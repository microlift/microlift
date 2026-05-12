use std::ffi::c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionArgumentType {
    None,
    SignedByte,
    SignedHalfword,
    SignedWord,
    SignedDoubleword,
    UnsignedByte,
    UnsignedHalfword,
    UnsignedWord,
    UnsignedDoubleword,
    SinglePrecisionFp,
    DoublePrecisionFp,
    Pointer,
    VariadicList,
}

pub trait ToFunctionArgumentType {
    const TYPE: FunctionArgumentType;
}

impl ToFunctionArgumentType for () {
    const TYPE: FunctionArgumentType = FunctionArgumentType::None;
}
impl ToFunctionArgumentType for bool {
    const TYPE: FunctionArgumentType = FunctionArgumentType::UnsignedByte;
}
impl ToFunctionArgumentType for i8 {
    const TYPE: FunctionArgumentType = FunctionArgumentType::SignedByte;
}
impl ToFunctionArgumentType for u8 {
    const TYPE: FunctionArgumentType = FunctionArgumentType::UnsignedByte;
}
impl ToFunctionArgumentType for i16 {
    const TYPE: FunctionArgumentType = FunctionArgumentType::SignedHalfword;
}
impl ToFunctionArgumentType for u16 {
    const TYPE: FunctionArgumentType = FunctionArgumentType::UnsignedHalfword;
}
impl ToFunctionArgumentType for i32 {
    const TYPE: FunctionArgumentType = FunctionArgumentType::SignedWord;
}
impl ToFunctionArgumentType for u32 {
    const TYPE: FunctionArgumentType = FunctionArgumentType::UnsignedWord;
}
impl ToFunctionArgumentType for i64 {
    const TYPE: FunctionArgumentType = FunctionArgumentType::SignedDoubleword;
}
impl ToFunctionArgumentType for u64 {
    const TYPE: FunctionArgumentType = FunctionArgumentType::UnsignedDoubleword;
}
impl ToFunctionArgumentType for usize {
    const TYPE: FunctionArgumentType = FunctionArgumentType::UnsignedDoubleword;
}
impl ToFunctionArgumentType for isize {
    const TYPE: FunctionArgumentType = FunctionArgumentType::SignedDoubleword;
}
impl ToFunctionArgumentType for f32 {
    const TYPE: FunctionArgumentType = FunctionArgumentType::SinglePrecisionFp;
}
impl ToFunctionArgumentType for f64 {
    const TYPE: FunctionArgumentType = FunctionArgumentType::DoublePrecisionFp;
}

// A blanket implementation for any raw pointer type.
impl<T> ToFunctionArgumentType for *const T {
    const TYPE: FunctionArgumentType = FunctionArgumentType::Pointer;
}
impl<T> ToFunctionArgumentType for *mut T {
    const TYPE: FunctionArgumentType = FunctionArgumentType::Pointer;
}

/// A trait that provides compile-time reflection for function pointers.
pub trait Function {
    /// Returns the function's argument and return types as a Vec.
    /// The return type is always the first element.
    fn get_argument_types(&self) -> Vec<FunctionArgumentType>;
    /// Returns the function pointer as a raw void pointer.
    fn as_ptr(&self) -> *mut c_void;
    /// Returns true if the function is C-variadic.
    fn is_variadic(&self) -> bool;
}

macro_rules! impl_function_trait {
    // Non-variadic case (0 or more arguments)
    ($($arg:ident),*) => {
        // Implementation for standard, safe `fn` pointers
        impl<Ret, $($arg),*> Function for fn($($arg),*) -> Ret
        where
            Ret: ToFunctionArgumentType,
            $($arg: ToFunctionArgumentType),*
        {
            fn get_argument_types(&self) -> Vec<FunctionArgumentType> {
                vec![
                    Ret::TYPE,
                    $($arg::TYPE),*
                ]
            }

            fn as_ptr(&self) -> *mut c_void {
                *self as *mut c_void
            }

            fn is_variadic(&self) -> bool {
                false
            }
        }

        // Implementation for `unsafe extern "C"` functions, which is common for C interop.
        impl<Ret, $($arg),*> Function for unsafe extern "C" fn($($arg),*) -> Ret
        where
            Ret: ToFunctionArgumentType,
            $($arg: ToFunctionArgumentType),*
        {
            fn get_argument_types(&self) -> Vec<FunctionArgumentType> {
                vec![
                    Ret::TYPE,
                    $($arg::TYPE),*
                ]
            }

            fn as_ptr(&self) -> *mut c_void {
                *self as *mut c_void
            }

            fn is_variadic(&self) -> bool {
                false
            }
        }

        impl<Ret, $($arg),*> ToFunctionArgumentType for Option<fn($($arg),*) -> Ret> {
            const TYPE: FunctionArgumentType = FunctionArgumentType::Pointer;
        }
        impl<Ret, $($arg),*> ToFunctionArgumentType for unsafe extern "C" fn($($arg),*) -> Ret {
            const TYPE: FunctionArgumentType = FunctionArgumentType::Pointer;
        }
        impl<Ret, $($arg),*> ToFunctionArgumentType for Option<unsafe extern "C" fn($($arg),*) -> Ret> {
            const TYPE: FunctionArgumentType = FunctionArgumentType::Pointer;
        }
    };
    // Variadic case (at least one argument)
    ($($arg:ident),+ ; variadic) => {
        impl<Ret, $($arg),+> Function for unsafe extern "C" fn($($arg),+, ...) -> Ret
        where
            Ret: ToFunctionArgumentType,
            $($arg: ToFunctionArgumentType),+
        {
            fn get_argument_types(&self) -> Vec<FunctionArgumentType> {
                let mut types = vec![
                    Ret::TYPE,
                    $($arg::TYPE),+
                ];
                types.push(FunctionArgumentType::VariadicList);
                types
            }

            fn as_ptr(&self) -> *mut c_void {
                *self as *mut c_void
            }

            fn is_variadic(&self) -> bool {
                true
            }
        }
    };
    // Variadic case (zero arguments)
    (; variadic) => {
        impl<Ret> Function for unsafe extern "C" fn(...) -> Ret
        where
            Ret: ToFunctionArgumentType,
        {
            fn get_argument_types(&self) -> Vec<FunctionArgumentType> {
                vec![
                    Ret::TYPE,
                    FunctionArgumentType::VariadicList,
                ]
            }

            fn as_ptr(&self) -> *mut c_void {
                *self as *mut c_void
            }

            fn is_variadic(&self) -> bool {
                true
            }
        }
    };
}

// functions with 0 to 8 arguments
impl_function_trait!();
impl_function_trait!(A);
impl_function_trait!(A, B);
impl_function_trait!(A, B, C);
impl_function_trait!(A, B, C, D);
impl_function_trait!(A, B, C, D, E);
impl_function_trait!(A, B, C, D, E, F);
impl_function_trait!(A, B, C, D, E, F, G);
impl_function_trait!(A, B, C, D, E, F, G, H);

// Variadic implementations
impl_function_trait!(; variadic);
impl_function_trait!(A ; variadic);
impl_function_trait!(A, B ; variadic);
impl_function_trait!(A, B, C ; variadic);
impl_function_trait!(A, B, C, D ; variadic);
impl_function_trait!(A, B, C, D, E ; variadic);
impl_function_trait!(A, B, C, D, E, F ; variadic);
impl_function_trait!(A, B, C, D, E, F, G ; variadic);
impl_function_trait!(A, B, C, D, E, F, G, H ; variadic);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    None,
    HasExecutionSideEffects,
}

#[derive(Debug, Clone)]
pub struct DynamicSymbol {
    pub name: &'static str,
    pub value: *mut c_void,
    pub arg_types: Vec<FunctionArgumentType>,
    pub is_ellipsis_function: bool,
    pub has_execution_side_effects: bool,
}

impl DynamicSymbol {
    /// Creates a new DynamicSymbol by inferring types from a function pointer.
    pub fn new<F: Function>(name: &'static str, func: F) -> Self {
        Self {
            name,
            value: func.as_ptr(),
            arg_types: func.get_argument_types(),
            is_ellipsis_function: func.is_variadic(),
            has_execution_side_effects: false,
        }
    }

    /// Creates a new DynamicSymbol with an additional tag.
    pub fn with_tag<F: Function>(tag: Tag, name: &'static str, func: F) -> Self {
        let mut sym = Self::new(name, func);
        if tag == Tag::HasExecutionSideEffects {
            sym.has_execution_side_effects = true;
        }
        sym
    }
}

pub struct DeferredSymbol {
    pub name: &'static str,
    pub value: *mut *mut c_void,
    pub arg_types: Vec<FunctionArgumentType>,
}

impl DeferredSymbol {
    /// Creates a new DeferredSymbol by inferring types from a mutable reference
    /// to a function pointer.
    pub fn new<F: Function>(name: &'static str, func: &'static mut F) -> Self {
        Self {
            name,
            // Cast the mutable reference to the function pointer to a void double pointer.
            value: func as *mut F as *mut *mut c_void,
            arg_types: func.get_argument_types(),
        }
    }
}

/// Convenience macro for creating a Function with proper casting.
#[macro_export]
macro_rules! trampoline_func {
    // Variadic (ellipsis) support
    ($func:path, $($arg:ty),*, ...) => {
        $func as unsafe extern "C" fn($($arg),*, ...) -> _
    };
    // No arguments
    ($func:path) => {
        $func as unsafe extern "C" fn() -> _
    };
    // Fixed arguments
    ($func:path, $($arg:ty),+) => {
        $func as unsafe extern "C" fn($($arg),*) -> _
    };
}
