use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::parse_macro_input;

/// A macro to pack multiple bytes into a u64 (big-endian order)
///
/// Takes a comma-separated list of u8 expressions and combines them into a single u64,
/// with the first byte becoming the most significant byte.
///
/// # Example
/// ```
/// let result = pack_bytes_u64!(0x12, 0x34, 0x56, 0x78);
/// assert_eq!(result, 0x12345678);
/// ```
#[proc_macro]
pub fn pack_bytes_u64(input: TokenStream) -> TokenStream {
    let bytes = parse_macro_input!(input with syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated).into_iter();
    let mut result = 0u64;

    for byte in bytes {
        let byte_str = byte.to_token_stream().to_string();
        let byte_val = if byte_str.starts_with("0b") {
            u64::from_str_radix(&byte_str[2..].replace("_", ""), 2).unwrap()
        } else {
            byte_str.parse::<u64>().unwrap()
        };
        result = (result << 8) | byte_val;
    }

    TokenStream::from(quote!(#result))
}

/// A macro to pack multiple bytes into a u32 (big-endian order)
///
/// Takes a comma-separated list of u8 expressions and combines them into a single u32,
/// with the first byte becoming the most significant byte.
///
/// # Example
/// ```
/// let result = pack_bytes_u32!(0x12, 0x34, 0x56, 0x78);
/// assert_eq!(result, 0x12345678);
/// ```
#[proc_macro]
pub fn pack_bytes_u32(input: TokenStream) -> TokenStream {
    let bytes = parse_macro_input!(input with syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated).into_iter();
    let mut result = 0u32;

    for byte in bytes {
        let byte_str = byte.to_token_stream().to_string();
        let byte_val = if byte_str.starts_with("0b") {
            u32::from_str_radix(&byte_str[2..].replace("_", ""), 2).unwrap()
        } else {
            byte_str.parse::<u32>().unwrap()
        };
        result = (result << 8) | byte_val;
    }

    TokenStream::from(quote!(#result))
}

/// A macro to pack multiple bytes into a u16 (big-endian order)
///
/// Takes a comma-separated list of u8 expressions and combines them into a single u16,
/// with the first byte becoming the most significant byte.
///
/// # Example
/// ```
/// let result = pack_bytes_u16!(0x12, 0x34);
/// assert_eq!(result, 0x1234);
/// ```
#[proc_macro]
pub fn pack_bytes_u16(input: TokenStream) -> TokenStream {
    let bytes = parse_macro_input!(input with syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated).into_iter();
    let mut result = 0u16;

    for byte in bytes {
        let byte_str = byte.to_token_stream().to_string();
        let byte_val = if byte_str.starts_with("0b") {
            u16::from_str_radix(&byte_str[2..].replace("_", ""), 2).unwrap()
        } else {
            byte_str.parse::<u16>().unwrap()
        };
        result = (result << 8) | byte_val;
    }

    TokenStream::from(quote!(#result))
}

/// A custom assertion macro that can be disabled at compile time to eliminate runtime overhead.
///
/// This macro only runs assertions when debug assertions are enabled and the `ASSERT_LEVEL`
/// is greater than or equal to 1. This allows for fine-grained control over which assertions
/// are active.
///
/// # Example
/// ```
/// assert_level!(x > 0); // Only runs in debug builds when ASSERT_LEVEL >= 1
/// ```
#[proc_macro]
pub fn assert_level(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    format!(
        r#"
        #[cfg(debug_assertions)]
        {{
            if ASSERT_LEVEL >= 1 {{
                assert!({}, "Assertion failed: {{}}", {});
            }}
        }}
    "#,
        input_str, input_str
    )
    .parse()
    .unwrap()
}

/// Converts a visual tetris piece representation into a byte array.
/// Each line must be exactly COLS characters of 1s and 0s.
/// The number of rows must be exactly ROW_CHUNK.
#[proc_macro]
pub fn piece_bytes(input: TokenStream) -> TokenStream {
    let input = input.to_string();
    let lines: Vec<&str> = input.split_whitespace().collect();
    if lines.len() != 4 {
        panic!("Only {} lines provided, expected 4", lines.len());
    }

    // Validate line lengths
    for line in &lines {
        if line.len() != 10 {
            panic!("Each line must be exactly 10 characters");
        }
    }

    // merge into one string
    let mut value = String::new();
    for line in &lines {
        value.push_str(line);
    }

    // read every 8 bits, turn to a u8 and add to a slice
    let mut bytes = Vec::with_capacity(5);
    for i in 0..5 {
        let byte =
            u8::from_str_radix(&value[i * 8..(i + 1) * 8], 2).expect("Invalid binary string");
        bytes.push(byte);
    }

    let result = quote! {
        [
            #(#bytes),*
        ]
    };

    result.into()
}

/// Converts a visual tetris piece representation into a u64.
#[proc_macro]
pub fn piece_u64(input: TokenStream) -> TokenStream {
    let input = input.to_string();
    let lines: Vec<&str> = input.split_whitespace().collect();
    if lines.len() != 4 {
        panic!("Only {} lines provided, expected 4", lines.len());
    }

    // Validate line lengths
    for line in &lines {
        if line.len() != 10 {
            panic!("Each line must be exactly 10 characters");
        }
    }

    // Convert to u64
    let mut result: u64 = 0;
    result |= (u64::from_str_radix(lines[0], 2).expect("Invalid binary string")) << 30;
    result |= (u64::from_str_radix(lines[1], 2).expect("Invalid binary string")) << 20;
    result |= (u64::from_str_radix(lines[2], 2).expect("Invalid binary string")) << 10;
    result |= u64::from_str_radix(lines[3], 2).expect("Invalid binary string");

    let result = quote! {
        #result
    };

    result.into()
}
