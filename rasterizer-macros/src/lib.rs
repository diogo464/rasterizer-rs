use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Error, Field, Result};

#[proc_macro_derive(Interpolate)]
pub fn derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let derive = parse_macro_input!(input as DeriveInput);

    match generate_impl(derive) {
        Ok(tokens) => tokens,
        Err(e) => e.to_compile_error(),
    }
    .into()
}

struct StructInfo {
    name: Ident,
    fields: Vec<Field>,
}

impl StructInfo {
    fn from_derive_input(derive: &DeriveInput) -> Result<Self> {
        let fields: Vec<Field> = if let syn::Data::Struct(syn::DataStruct {
            fields: syn::Fields::Named(syn::FieldsNamed { ref named, .. }),
            ..
        }) = derive.data
        {
            named.iter().map(|f| f.clone()).collect()
        } else {
            return Err(Error::new(
                Span::call_site(),
                "Only sturct with named fields are allowed by the builder macro",
            ));
        };

        Ok(Self {
            name: derive.ident.clone(),
            fields,
        })
    }
}

fn generate_field_interpolation(field: &Field) -> TokenStream {
    let name = field
        .ident
        .as_ref()
        .ok_or(Error::new(Span::call_site(), "Field must have a name"))
        .unwrap();
    let ty = &field.ty;
    quote! {
        #name : #ty::interpolate(&v0.#name, &v1.#name, &v2.#name, r0, r1, r2)
    }
}

fn generate_impl(derive: DeriveInput) -> Result<TokenStream> {
    let struct_info = StructInfo::from_derive_input(&derive)?;

    let struct_name = &struct_info.name;

    let field_assignements: Vec<TokenStream> = struct_info
        .fields
        .iter()
        .map(|f| generate_field_interpolation(f))
        .collect();

    Ok(quote! {
        impl Interpolate for #struct_name {
            fn interpolate(v0 : &Self, v1 : &Self, v2 : &Self, r0 : f32, r1 : f32, r2 : f32) -> Self {
                Self {
                    #(#field_assignements),*
                }
            }
        }
    })
}
