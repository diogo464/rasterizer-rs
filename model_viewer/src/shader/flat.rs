use glam::Vec4;
use rasterizer::FragmentShader;

use crate::texture::Texture;

use super::{ProjViewModel, StandardShaderData};

pub struct FlatTextureFragmentShader {
    image: Texture,
}
impl FlatTextureFragmentShader {
    pub fn new(img: Texture) -> Self {
        Self { image: img }
    }
}
impl FragmentShader for FlatTextureFragmentShader {
    type Uniform = ProjViewModel;
    type SharedData = StandardShaderData;

    fn fragment(&self, shared: &Self::SharedData, _uniform: &Self::Uniform) -> Vec4 {
        self.image.sample(shared.texture_coords)
    }
}
