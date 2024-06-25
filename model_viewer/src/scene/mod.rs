mod camera;
mod diablo3;
mod pbr;

use glam::Vec3;
use rasterizer::Rasterizer;

pub use camera::Camera;
pub use diablo3::DiabloScene;
pub use pbr::PBRScene;

pub trait Scene {
    fn render(&mut self, model_viewer: &mut ModelViewer);
}

pub struct ModelViewer {
    pub camera: Camera,
    pub light_position: Vec3,
    pub rasterizer: Rasterizer,
}
