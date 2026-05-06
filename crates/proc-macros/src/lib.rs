use proc_macro::TokenStream;
use quote::quote;
use syn::{Ident, ItemFn, parse_macro_input};

/// Helper: convert a fallible macro body into a compile_error on failure.
fn try_macro(f: impl FnOnce() -> Result<TokenStream, String>) -> TokenStream {
    match f() {
        Ok(ts) => ts,
        Err(msg) => {
            let msg = msg.as_str();
            quote! { compile_error!(#msg); }.into()
        }
    }
}

/// Converts a visual tetris piece representation into a byte array.
/// Each line must be exactly COLS characters of 1s and 0s.
/// The number of rows must be exactly ROW_CHUNK.
#[proc_macro]
pub fn piece_bytes(input: TokenStream) -> TokenStream {
    try_macro(|| {
        let input = input.to_string();
        let lines: Vec<&str> = input.split_whitespace().collect();
        if lines.len() != 4 {
            return Err(format!("expected 4 lines, got {}", lines.len()));
        }

        for line in &lines {
            if line.len() != 10 {
                return Err(format!(
                    "each line must be exactly 10 characters, got {} in '{line}'",
                    line.len()
                ));
            }
        }

        let mut value = String::new();
        for line in &lines {
            value.push_str(line);
        }

        let mut bytes = Vec::with_capacity(5);
        for i in 0..5 {
            let byte = u8::from_str_radix(&value[i * 8..(i + 1) * 8], 2)
                .map_err(|e| format!("invalid binary string: {e}"))?;
            bytes.push(byte);
        }

        Ok(quote! { [#(#bytes),*] }.into())
    })
}

/// Converts a visual tetris piece representation into a u64.
#[proc_macro]
pub fn piece_u64(input: TokenStream) -> TokenStream {
    try_macro(|| {
        let input = input.to_string();
        let lines: Vec<&str> = input.split_whitespace().collect();
        if lines.len() != 4 {
            return Err(format!("expected 4 lines, got {}", lines.len()));
        }

        for line in &lines {
            if line.len() != 10 {
                return Err(format!(
                    "each line must be exactly 10 characters, got {} in '{line}'",
                    line.len()
                ));
            }
        }

        let mut result: u64 = 0;
        for (i, line) in lines.iter().enumerate() {
            let val = u64::from_str_radix(line, 2)
                .map_err(|e| format!("invalid binary in line {i}: {e}"))?;
            result |= val << (30 - i * 10);
        }

        Ok(quote! { #result }.into())
    })
}

/// Converts a visual representation of a Tetris piece into an array of u32s, where each u32 represents a column.
///
/// This macro takes a 4x3 grid of 1s and 0s as input, where 1s represent filled cells and 0s represent empty cells.
/// Each column in the input grid maps to a u32 in the output array, with the bits shifted to the high end.
///
/// # Arguments
///
/// * Input must be exactly 4 rows of equal length, containing only 1s and 0s
///
/// # Returns
///
/// * An array of u32s, where each u32 represents a column from the input grid
///
/// # Examples
///
/// ```
/// use proc_macros::piece_u32_cols;
///
/// // Create an L-shaped piece:
/// let l_piece = piece_u32_cols! {
///     111  // Maps to [0b1100_0000..., 0b1000_0000..., 0b1000_0000...]
///     100
///     000
///     000
/// };
/// assert_eq!(l_piece, [
///     0b1100_0000_0000_0000_0000_0000_0000_0000,
///     0b1000_0000_0000_0000_0000_0000_0000_0000,
///     0b1000_0000_0000_0000_0000_0000_0000_0000
/// ]);
///
/// // Create a T-shaped piece:
/// let t_piece = piece_u32_cols! {
///     111  // Maps to [0b1000_0000..., 0b1100_0000..., 0b1000_0000...]
///     010
///     000
///     000
/// };
/// assert_eq!(t_piece, [
///     0b1000_0000_0000_0000_0000_0000_0000_0000,
///     0b1100_0000_0000_0000_0000_0000_0000_0000,
///     0b1000_0000_0000_0000_0000_0000_0000_0000
/// ]);
/// ```
#[proc_macro]
pub fn piece_u32_cols(input: TokenStream) -> TokenStream {
    try_macro(|| {
        let input = input.to_string();
        let lines: Vec<&str> = input.split_whitespace().collect();
        if lines.len() != 4 {
            return Err(format!("expected 4 lines, got {}", lines.len()));
        }

        let first_row_len = lines[0].len();
        if !lines.iter().all(|line| line.len() == first_row_len) {
            return Err("all lines must have the same length".to_string());
        }
        let num_cols = first_row_len;

        let mut cols = vec![0u32; num_cols];
        for (col_idx, col) in cols.iter_mut().enumerate().take(num_cols) {
            let mut col_val = 0u32;
            for (row_idx, line) in lines.iter().enumerate() {
                let bit = line.chars().nth(col_idx).unwrap();
                if bit == '1' {
                    col_val |= 1u32 << (31 - row_idx);
                }
            }
            *col = col_val;
        }

        Ok(quote! { [#(#cols),*] }.into())
    })
}

/// Conditional inline attribute that respects the `never-inline` feature flag.
///
/// This attribute macro allows fine-grained control over inlining behavior for profiling
/// and benchmarking. When the `never-inline` feature is enabled, all functions marked with
/// this attribute become `#[inline(never)]`, making them visible in profilers.
///
/// # Supported Forms
///
/// - `#[inline_conditioned]` - Applies `#[inline]` normally, `#[inline(never)]` with feature
/// - `#[inline_conditioned(always)]` - Applies `#[inline(always)]` normally, `#[inline(never)]` with feature
/// - `#[inline_conditioned(never)]` - Always applies `#[inline(never)]`, regardless of feature
///
/// # Examples
///
/// ```rust,ignore
/// #[inline_conditioned(always)]
/// fn hot_path() {
///     // This function is aggressively inlined normally,
///     // but visible in profilers when built with --features never-inline
/// }
///
/// #[inline_conditioned]
/// fn helper() {
///     // Standard inline hint that can be disabled for profiling
/// }
/// ```
#[proc_macro_attribute]
pub fn inline_conditioned(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_fn = parse_macro_input!(item as ItemFn);

    let output = if attr.is_empty() {
        quote! {
            #[cfg_attr(not(feature = "never-inline"), inline)]
            #[cfg_attr(feature = "never-inline", inline(never))]
            #item_fn
        }
    } else {
        match parse_macro_input!(attr as Ident).to_string().as_str() {
            "always" => quote! {
                #[cfg_attr(not(feature = "never-inline"), inline(always))]
                #[cfg_attr(feature = "never-inline", inline(never))]
                #item_fn
            },
            "never" => quote! {
                #[inline(never)]
                #item_fn
            },
            other => {
                return syn::Error::new_spanned(
                    item_fn.sig.ident,
                    format!(
                        "invalid inline hint `{}`; expected `always` or `never`",
                        other
                    ),
                )
                .to_compile_error()
                .into();
            }
        }
    };

    output.into()
}
