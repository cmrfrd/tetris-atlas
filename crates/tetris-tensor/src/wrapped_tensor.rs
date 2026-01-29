use anyhow::Result as AnyResult;
use candle_core::{DType, Device, Shape, Tensor};

pub(crate) mod sealed {
    pub trait Sealed {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeDim {
    Any,
    Dim(usize),
}

/// Sealed trait to be implemented only for single-field tuple structs
/// that wrap a `candle_core::Tensor`. Use the macro `impl_wrapped_tensor!`
/// to implement this trait for your wrapper type.
pub trait WrappedTensor: sealed::Sealed {
    const RANK: usize;
    type ShapeSpec;
    type ShapeTuple;

    fn inner(&self) -> &Tensor;
    fn inner_mut(&mut self) -> &mut Tensor;

    /// Required dtype for this wrapper.
    fn expected_dtype() -> DType;

    /// Shape validator for the inner tensor.
    fn shape_ok(tensor: &Tensor) -> AnyResult<()>;

    /// Compile-time shape specification provided by the macro.
    fn shape_spec(&self) -> Self::ShapeSpec;

    /// Compile-time shape tuple provided by the macro.
    fn shape_tuple(&self) -> Self::ShapeTuple;

    /// Actual runtime tensor shape
    fn tensor_shape(&self) -> &Shape {
        self.inner().shape()
    }

    fn device(&self) -> &Device {
        self.inner().device()
    }

    fn dtype(&self) -> DType {
        self.inner().dtype()
    }
}

#[macro_export]
macro_rules! impl_wrapped_tensor {
    ($ty:ty, dtype = $dtype:expr, shape_spec = ( $($spec:expr),* $(,)? )) => {
        impl $crate::wrapped_tensor::sealed::Sealed for $ty {}
        impl $crate::wrapped_tensor::WrappedTensor for $ty {
            const RANK: usize = $crate::__count_tuple_elements!( $($spec),* );

            type ShapeSpec = $crate::__tensors_type_tuple_by_specs!( $crate::wrapped_tensor::ShapeDim ; $($spec),* );
            type ShapeTuple = $crate::__tensors_type_tuple_by_specs!( usize ; $($spec),* );

            #[proc_macros::inline_conditioned]
            fn inner(&self) -> &candle_core::Tensor {
                &self.0
            }

            #[proc_macros::inline_conditioned]
            fn inner_mut(&mut self) -> &mut candle_core::Tensor {
                &mut self.0
            }

            #[proc_macros::inline_conditioned]
            fn expected_dtype() -> candle_core::DType {
                $dtype
            }

            #[proc_macros::inline_conditioned]
            fn shape_ok(tensor: &candle_core::Tensor) -> ::anyhow::Result<()> {
                let expected: &[$crate::wrapped_tensor::ShapeDim] = &[$($spec),*];
                let actual = tensor.shape().dims();
                if actual.len() != expected.len() {
                    return Err(::anyhow::anyhow!(
                        "rank mismatch: got {}, expected {}",
                        actual.len(),
                        expected.len()
                    ));
                }
                for (axis, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                    match *e {
                        $crate::wrapped_tensor::ShapeDim::Any => {},
                        $crate::wrapped_tensor::ShapeDim::Dim(n) => {
                            if *a != n {
                                return Err(::anyhow::anyhow!(
                                    "dim {} mismatch: got {}, expected {}",
                                    axis, a, n
                                ));
                            }
                        }
                    }
                }
                Ok(())
            }

            #[proc_macros::inline_conditioned]
            fn shape_spec(&self) -> Self::ShapeSpec { ( $($spec),* ) }

            #[proc_macros::inline_conditioned]
            fn shape_tuple(&self) -> Self::ShapeTuple {
                let sized_dims: [usize; $crate::__count_tuple_elements!( $($spec),* )] = self.tensor_shape().dims().try_into().unwrap();
                sized_dims.into()
            }
        }

        impl std::ops::Deref for $ty {
            type Target = candle_core::Tensor;

            #[proc_macros::inline_conditioned]
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl Into<candle_core::Tensor> for $ty {
            #[proc_macros::inline_conditioned]
            fn into(self) -> candle_core::Tensor {
                self.0
            }
        }

        impl std::convert::TryFrom<candle_core::Tensor> for $ty {
            type Error = ::anyhow::Error;

            fn try_from(tensor: candle_core::Tensor) -> ::anyhow::Result<Self> {
                let expected = <$ty as $crate::wrapped_tensor::WrappedTensor>::expected_dtype();
                if tensor.dtype() != expected {
                    return Err(::anyhow::anyhow!(
                        "Tensor must be {:?}, got {:?}",
                        expected,
                        tensor.dtype()
                    ));
                }

                // Shape check
                <$ty as $crate::wrapped_tensor::WrappedTensor>::shape_ok(&tensor)?;

                Ok(Self(tensor))
            }
        }
    };
}

#[macro_export]
macro_rules! __count_tuple_elements {
    () => { 0usize };
    ($head:expr $(, $tail:expr)*) => { 1usize + $crate::__count_tuple_elements!($($tail),*) };
}

#[macro_export]
macro_rules! __tensors_type_tuple_by_specs {
    ($type:ty ; $a:expr) => {
        ($type,)
    };
    ($type:ty ; $a:expr, $b:expr) => {
        ($type, $type)
    };
    ($type:ty ; $a:expr, $b:expr, $c:expr) => {
        ($type, $type, $type)
    };
    ($type:ty ; $a:expr, $b:expr, $c:expr, $d:expr) => {
        ($type, $type, $type, $type)
    };
    ($type:ty ; $a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {
        ($type, $type, $type, $type, $type)
    };
}
