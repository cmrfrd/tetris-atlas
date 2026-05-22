//! Compile-time codegen for conditional inlining.
//! Zero external deps — uses only the built-in `proc_macro` crate.

use proc_macro::TokenStream;

/// Parse a string of Rust code back into a TokenStream.
fn code(src: impl AsRef<str>) -> TokenStream {
    src.as_ref().parse().expect("generated invalid Rust")
}

/// Conditional inline hint that respects the `never-inline` feature flag.
///
/// | input                           | normal build        | `--features never-inline` |
/// |---------------------------------|---------------------|---------------------------|
/// | `#[inline_conditioned]`         | `#[inline]`         | `#[inline(never)]`        |
/// | `#[inline_conditioned(always)]` | `#[inline(always)]` | `#[inline(never)]`        |
/// | `#[inline_conditioned(never)]`  | `#[inline(never)]`  | `#[inline(never)]`        |
///
/// # Examples
///
/// ```
/// use proc_macros::inline_conditioned;
///
/// #[inline_conditioned(always)]
/// fn hot_path(x: u32) -> u32 { x + 1 }
///
/// #[inline_conditioned]
/// fn helper() -> u32 { 7 }
///
/// #[inline_conditioned(never)]
/// fn cold_path() -> u32 { 13 }
///
/// assert_eq!(hot_path(2), 3);
/// assert_eq!(helper(), 7);
/// assert_eq!(cold_path(), 13);
/// ```
#[proc_macro_attribute]
pub fn inline_conditioned(attr: TokenStream, item: TokenStream) -> TokenStream {
    let hint = match attr.to_string().trim() {
        "" => "inline",
        "always" => "inline(always)",
        "never" => return prepend("#[inline(never)]", item),
        bad => return code(format!(r#"compile_error!("invalid inline hint `{bad}`");"#)),
    };
    prepend(
        &format!(
            "#[cfg_attr(not(feature = \"never-inline\"), {hint})]\n\
             #[cfg_attr(feature = \"never-inline\", inline(never))]"
        ),
        item,
    )
}

fn prepend(prefix: &str, item: TokenStream) -> TokenStream {
    let mut out: TokenStream = prefix.parse().unwrap();
    out.extend(item);
    out
}
