use glam::{Vec3, Vec4};
use rasterizer::FragmentShader;

use super::{ProjViewModel, StandardShaderData};

#[derive(Default)]
pub struct NormalShadingFragmentShader {}
impl NormalShadingFragmentShader {
    pub fn new() -> Self {
        Self::default()
    }
}
impl FragmentShader for NormalShadingFragmentShader {
    type Uniform = ProjViewModel;
    type SharedData = StandardShaderData;

    fn fragment(&self, shared: &Self::SharedData, _uniform: &Self::Uniform) -> Vec4 {
        let shading = shared.normal.dot(Vec3::new(0.0, 1.0, 0.0)).min(0.8);
        Vec4::new(shading, shading, shading, 1.0)
    }
}
