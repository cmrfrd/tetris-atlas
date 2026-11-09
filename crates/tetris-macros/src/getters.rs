use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Ident, LitStr, Member, Meta};

pub(crate) fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "Getters supports structs only",
        ));
    };
    let mut methods = Vec::new();
    for (index, field) in data.fields.iter().enumerate() {
        let mut attributes = field.attrs.iter().filter(|a| a.path().is_ident("getter"));
        let Some(attribute) = attributes.next() else {
            continue;
        };
        if let Some(duplicate) = attributes.next() {
            return Err(syn::Error::new_spanned(
                duplicate,
                "duplicate getter attribute",
            ));
        }
        let mut copy = false;
        let mut name = None;
        match &attribute.meta {
            Meta::Path(_) => {}
            Meta::List(_) => attribute.parse_nested_meta(|meta| {
                if meta.path.is_ident("copy") {
                    if copy {
                        return Err(meta.error("duplicate copy option"));
                    }
                    copy = true;
                } else if meta.path.is_ident("name") {
                    if name.is_some() {
                        return Err(meta.error("duplicate name option"));
                    }
                    name = Some(meta.value()?.parse::<LitStr>()?.parse::<Ident>()?);
                } else {
                    return Err(meta.error("expected copy or name = \"method_name\""));
                }
                Ok(())
            })?,
            Meta::NameValue(_) => {
                return Err(syn::Error::new_spanned(
                    attribute,
                    "expected #[getter] or #[getter(...)]",
                ));
            }
        }
        let name = name.or_else(|| field.ident.clone()).ok_or_else(|| {
            syn::Error::new_spanned(
                attribute,
                "tuple field getters require name = \"method_name\"",
            )
        })?;
        let member = field
            .ident
            .clone()
            .map(Member::Named)
            .unwrap_or_else(|| Member::Unnamed(index.into()));
        let ty = &field.ty;
        let docs = field.attrs.iter().filter(|a| a.path().is_ident("doc"));
        let inline = crate::conditioned_inline(quote!(inline(always)));
        let (return_type, value) = if copy {
            (quote!(#ty), quote!(self.#member))
        } else {
            (quote!(&#ty), quote!(&self.#member))
        };
        methods.push(quote! {
            #(#docs)*
            #inline
            #[must_use]
            pub const fn #name(&self) -> #return_type {
                #value
            }
        });
    }
    let name = &input.ident;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics #name #type_generics #where_clause {
            #(#methods)*
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn rejects_non_structs() {
        let input = parse_quote!(
            enum State {
                Empty,
            }
        );
        assert!(
            expand(&input)
                .expect_err("invalid getter input must be rejected in rejects_non_structs")
                .to_string()
                .contains("structs only")
        );
    }

    #[test]
    fn rejects_unknown_options() {
        let input = parse_quote!(
            struct State {
                #[getter(mutable)]
                value: u8,
            }
        );
        assert!(
            expand(&input)
                .expect_err("invalid getter input must be rejected in rejects_unknown_options")
                .to_string()
                .contains("expected copy or name")
        );
    }

    #[test]
    fn rejects_duplicate_attributes() {
        let input = parse_quote!(
            struct State {
                #[getter]
                #[getter(copy)]
                value: u8,
            }
        );
        assert!(
            expand(&input)
                .expect_err("invalid getter input must be rejected in rejects_duplicate_attributes")
                .to_string()
                .contains("duplicate getter attribute")
        );
    }

    #[test]
    fn rejects_unnamed_tuple_getters() {
        let input = parse_quote!(
            struct State(#[getter] u8);
        );
        assert!(
            expand(&input)
                .expect_err(
                    "invalid getter input must be rejected in rejects_unnamed_tuple_getters"
                )
                .to_string()
                .contains("tuple field getters require name")
        );
    }

    #[test]
    fn rejects_invalid_method_names() {
        let input = parse_quote!(
            struct State {
                #[getter(name = "not a name")]
                value: u8,
            }
        );
        assert!(expand(&input).is_err());
    }
}
