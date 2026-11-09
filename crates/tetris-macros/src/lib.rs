//! Field accessors and conditional inlining.

use proc_macro::TokenStream;
use quote::quote;

mod getters;

/// Generates public, const, read-only accessors for fields marked with `#[getter]`.
///
/// Accessors borrow the field by default. Use `#[getter(copy)]` to return its value,
/// or `#[getter(name = "method_name")]` to choose a name. Tuple fields require a name.
/// Field documentation is copied to the accessor. Generated methods are `#[must_use]`
/// and use the same conditional inlining as `#[inline_conditioned(always)]`.
///
/// ```
/// use tetris_macros::Getters;
///
/// #[derive(Getters)]
/// struct State {
///     #[getter(copy)]
///     count: u8,
///     #[getter(name = "values")]
///     data: [u8; 2],
/// }
///
/// const STATE: State = State { count: 2, data: [3, 4] };
/// const COUNT: u8 = STATE.count();
/// const VALUES: &[u8; 2] = STATE.values();
/// assert_eq!(COUNT, 2);
/// assert_eq!(VALUES, &[3, 4]);
/// ```
///
/// Borrowed accessors do not permit mutation:
///
/// ```compile_fail,E0596
/// use tetris_macros::Getters;
/// #[derive(Getters)]
/// struct State { #[getter] text: String }
/// let mut state = State { text: String::new() };
/// state.text().push('x');
/// ```
///
/// Copy accessors require a `Copy` field:
///
/// ```compile_fail,E0507
/// use tetris_macros::Getters;
/// #[derive(Getters)]
/// struct State { #[getter(copy)] text: String }
/// ```
#[proc_macro_derive(Getters, attributes(getter))]
pub fn derive_getters(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    getters::expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

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
/// use tetris_macros::inline_conditioned;
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
        "" => quote!(inline),
        "always" => quote!(inline(always)),
        "never" => return prepend("#[inline(never)]", item),
        bad => return code(format!(r#"compile_error!("invalid inline hint `{bad}`");"#)),
    };
    let mut out: TokenStream = conditioned_inline(hint).into();
    out.extend(item);
    out
}

fn conditioned_inline(hint: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    quote! {
        #[cfg_attr(not(feature = "never-inline"), #hint)]
        #[cfg_attr(feature = "never-inline", inline(never))]
    }
}

fn prepend(prefix: &str, item: TokenStream) -> TokenStream {
    let mut out: TokenStream = prefix
        .parse()
        .expect("generated Rust tokens must parse in prepend");
    out.extend(item);
    out
}
