use std::ptr;
use std::sync::{Arc, Mutex};

use opencl3::memory::{Buffer, ClMem, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_mem, CL_BLOCKING};

use crate::device::{DeviceArc, OpenCLDevice};
use crate::error::{CleError, Result};
use crate::types::{DType, MType};

// ── GPU memory handle ────────────────────────────────────────────────────────

/// Erased GPU memory — wraps `Mutex<Buffer<u8>>` for buffers.
/// Mutex provides interior mutability needed by `enqueue_write_buffer(&mut Buffer<T>)`.
pub enum GpuMemory {
    Buffer(Mutex<Buffer<u8>>),
}

pub type GpuMemPtr = Arc<GpuMemory>;

impl GpuMemory {
    /// Return the raw `cl_mem` handle for use as a kernel argument.
    pub fn cl_mem(&self) -> cl_mem {
        match self {
            GpuMemory::Buffer(m) => {
                let guard = m.lock().unwrap();
                guard.get()
            }
        }
    }
}

// Safety: opencl3 Buffer<u8> is Send + Sync; Mutex provides sync.
unsafe impl Send for GpuMemory {}
unsafe impl Sync for GpuMemory {}

// ── Backend trait ────────────────────────────────────────────────────────────

pub trait Backend: Send + Sync {
    fn allocate_memory(
        &self,
        device: &DeviceArc,
        region: [usize; 3],
        dtype: DType,
        mtype: MType,
    ) -> Result<GpuMemPtr>;

    fn write_memory(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        region: [usize; 3],
        origin: [usize; 3],
        host: *const u8,
        byte_size: usize,
    ) -> Result<()>;

    fn read_memory(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        region: [usize; 3],
        origin: [usize; 3],
        host: *mut u8,
        byte_size: usize,
    ) -> Result<()>;

    fn copy_memory(
        &self,
        device: &DeviceArc,
        src: &GpuMemPtr,
        dst: &GpuMemPtr,
        byte_size: usize,
    ) -> Result<()>;

    fn set_memory(&self, device: &DeviceArc, mem: &GpuMemPtr, value: f32, dtype: DType, element_count: usize) -> Result<()>;

    fn build_program(&self, device: &DeviceArc, source: &str) -> Result<Arc<Program>>;

    fn execute_kernel(
        &self,
        device: &DeviceArc,
        source: &str,
        kernel_name: &str,
        global_range: [usize; 3],
        local_range: [usize; 3],
        args: &[KernelArg],
    ) -> Result<()>;

    fn preamble(&self) -> &'static str;
}

// ── Kernel argument types ────────────────────────────────────────────────────

pub enum KernelArg {
    Mem(GpuMemPtr),
    Float(f32),
    Int(i32),
    Uint(u32),
    SizeT(usize),
}

// ── OpenCL backend implementation ────────────────────────────────────────────

pub struct OpenCLBackend;

impl OpenCLBackend {
    fn cast(device: &DeviceArc) -> &OpenCLDevice {
        // Safety: we only ever place OpenCLDevice behind DeviceArc in enumerate_opencl_devices
        unsafe { &*(Arc::as_ptr(device) as *const OpenCLDevice) }
    }
}

impl Backend for OpenCLBackend {
    fn preamble(&self) -> &'static str {
        include_str!("../kernels/preamble.cl")
    }

    fn allocate_memory(
        &self,
        device: &DeviceArc,
        region: [usize; 3],
        dtype: DType,
        _mtype: MType,
    ) -> Result<GpuMemPtr> {
        let ocl = Self::cast(device);
        let byte_count = region[0] * region[1] * region[2] * dtype.byte_size();
        // Safety: Buffer::create is an FFI call; args are valid
        let buf = unsafe {
            Buffer::<u8>::create(&ocl.context, CL_MEM_READ_WRITE, byte_count, ptr::null_mut())
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(Arc::new(GpuMemory::Buffer(Mutex::new(buf))))
    }

    fn write_memory(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        _region: [usize; 3],
        _origin: [usize; 3],
        host: *const u8,
        byte_size: usize,
    ) -> Result<()> {
        let ocl = Self::cast(device);
        let GpuMemory::Buffer(mutex_buf) = mem.as_ref();
        let mut guard = mutex_buf.lock().unwrap();
        // Safety: host pointer is valid for byte_size bytes (caller's contract)
        let slice = unsafe { std::slice::from_raw_parts(host, byte_size) };
        let _evt = unsafe {
            ocl.queue
                .enqueue_write_buffer(&mut *guard, CL_BLOCKING, 0, slice, &[])
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(())
    }

    fn read_memory(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        _region: [usize; 3],
        _origin: [usize; 3],
        host: *mut u8,
        byte_size: usize,
    ) -> Result<()> {
        let ocl = Self::cast(device);
        let GpuMemory::Buffer(mutex_buf) = mem.as_ref();
        let guard = mutex_buf.lock().unwrap();
        // Safety: host pointer is valid for byte_size bytes
        let slice = unsafe { std::slice::from_raw_parts_mut(host, byte_size) };
        let _evt = unsafe {
            ocl.queue
                .enqueue_read_buffer(&guard, CL_BLOCKING, 0, slice, &[])
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(())
    }

    fn copy_memory(
        &self,
        device: &DeviceArc,
        src: &GpuMemPtr,
        dst: &GpuMemPtr,
        byte_size: usize,
    ) -> Result<()> {
        let ocl = Self::cast(device);
        let GpuMemory::Buffer(src_mutex) = src.as_ref();
        let GpuMemory::Buffer(dst_mutex) = dst.as_ref();
        let src_guard = src_mutex.lock().unwrap();
        let mut dst_guard = dst_mutex.lock().unwrap();
        let _evt = unsafe {
            ocl.queue
                .enqueue_copy_buffer(&*src_guard, &mut *dst_guard, 0, 0, byte_size, &[])
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(())
    }

    fn set_memory(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        value: f32,
        dtype: DType,
        element_count: usize,
    ) -> Result<()> {
        let ocl = Self::cast(device);
        let GpuMemory::Buffer(mutex_buf) = mem.as_ref();
        let mut guard = mutex_buf.lock().unwrap();
        let byte_size = element_count * dtype.byte_size();

        macro_rules! fill_typed {
            ($T:ty) => {{
                let v = value as $T;
                let data: Vec<$T> = vec![v; element_count];
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_size)
                };
                unsafe {
                    ocl.queue
                        .enqueue_write_buffer(&mut *guard, CL_BLOCKING, 0, bytes, &[])
                        .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
                };
            }};
        }

        match dtype {
            DType::Float => fill_typed!(f32),
            DType::Int8 => fill_typed!(i8),
            DType::Uint8 => fill_typed!(u8),
            DType::Int16 => fill_typed!(i16),
            DType::Uint16 => fill_typed!(u16),
            DType::Int32 => fill_typed!(i32),
            DType::Uint32 => fill_typed!(u32),
            _ => return Err(CleError::InvalidDtype),
        }
        Ok(())
    }

    fn build_program(&self, device: &DeviceArc, source: &str) -> Result<Arc<Program>> {
        let source_hash = crate::cache::DiskCache::hash(source);
        let dev = Self::cast(device);

        // 1. Check in-memory LRU cache first
        if let Some(prog) = device.get_program_from_cache(&source_hash) {
            return Ok(prog);
        }

        // 2. Compile from source
        let program = Program::create_and_build_from_source(&dev.context, source, "")
            .map_err(|e| CleError::OpenCL(format!(
                "Build failed: {:?}\nSource (first 500 chars):\n{}",
                e,
                &source[..source.len().min(500)]
            )))?;
        let prog_arc = Arc::new(program);

        device.add_program_to_cache(source_hash, prog_arc.clone());
        Ok(prog_arc)
    }

    fn execute_kernel(
        &self,
        device: &DeviceArc,
        source: &str,
        kernel_name: &str,
        global_range: [usize; 3],
        local_range: [usize; 3],
        args: &[KernelArg],
    ) -> Result<()> {
        let ocl = Self::cast(device);
        let program = self.build_program(device, source)?;

        let kernel = opencl3::kernel::Kernel::create(&program, kernel_name)
            .map_err(|e| CleError::OpenCL(format!("Kernel '{}' create failed: {:?}", kernel_name, e)))?;

        // Set kernel arguments
        for (i, arg) in args.iter().enumerate() {
            match arg {
                KernelArg::Mem(mem) => {
                    let handle: cl_mem = mem.cl_mem();
                    unsafe {
                        kernel.set_arg(i as u32, &handle)
                            .map_err(|e| CleError::OpenCL(format!("set_arg({}) mem failed: {:?}", i, e)))?
                    };
                }
                KernelArg::Float(v) => {
                    unsafe {
                        kernel.set_arg(i as u32, v)
                            .map_err(|e| CleError::OpenCL(format!("set_arg({}) float failed: {:?}", i, e)))?
                    };
                }
                KernelArg::Int(v) => {
                    unsafe {
                        kernel.set_arg(i as u32, v)
                            .map_err(|e| CleError::OpenCL(format!("set_arg({}) int failed: {:?}", i, e)))?
                    };
                }
                KernelArg::Uint(v) => {
                    unsafe {
                        kernel.set_arg(i as u32, v)
                            .map_err(|e| CleError::OpenCL(format!("set_arg({}) uint failed: {:?}", i, e)))?
                    };
                }
                KernelArg::SizeT(v) => {
                    let v_u64 = *v as u64;
                    unsafe {
                        kernel.set_arg(i as u32, &v_u64)
                            .map_err(|e| CleError::OpenCL(format!("set_arg({}) sizet failed: {:?}", i, e)))?
                    };
                }
            }
        }

        let work_dim = if global_range[2] > 1 { 3 } else if global_range[1] > 1 { 2 } else { 1 };
        let local_ptr = if local_range[0] == 0 { ptr::null() } else { local_range.as_ptr() };

        let _evt = unsafe {
            ocl.queue
                .enqueue_nd_range_kernel(kernel.get(), work_dim, ptr::null(), global_range.as_ptr(), local_ptr, &[])
                .map_err(|e| CleError::OpenCL(format!("enqueue_nd_range_kernel '{}' failed: {:?}", kernel_name, e)))?
        };

        Ok(())
    }
}
