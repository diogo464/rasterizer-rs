use glm::Vec4;
use image::{DynamicImage, GenericImageView};
use nalgebra_glm as glm;

pub trait Texture {
    fn color_at(&self, u: f32, v: f32) -> Vec4;
}

impl Texture for DynamicImage {
    fn color_at(&self, u: f32, v: f32) -> Vec4 {
        let x = (u * self.width() as f32) as u32;
        let y = ((1.0 - v) * self.height() as f32) as u32;
        if let Some(c) = self.as_rgb8() {
            let p = c.get_pixel(x, y);
            Vec4::new(
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
                1.0,
            )
        } else if let Some(c) = self.as_rgba8() {
            let p = c.get_pixel(x, y);
            Vec4::new(
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
                p[3] as f32 / 255.0,
            )
        } else {
            panic!("Invalid pixel mode");
        }
    }
}
