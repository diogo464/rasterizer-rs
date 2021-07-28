pub mod rasterizer;
pub use rasterizer::*;
pub use rasterizer_macros::Interpolate;

pub mod math_prelude {
    pub use glam::{Mat2, Mat3, Mat4, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
}
