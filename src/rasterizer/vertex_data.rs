use glm::{Vec2, Vec3};
use nalgebra_glm as glm;

#[derive(Debug)]
pub struct VertexData {
    pub position: Vec3,
    pub texture: Option<Vec2>,
    pub normal: Option<Vec3>,
}

impl VertexData {
    pub fn new(position: Vec3, texture: Vec2, normal: Vec3) -> Self {
        Self {
            position,
            texture: Some(texture),
            normal: Some(normal),
        }
    }

    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            texture: None,
            normal: None,
        }
    }
}
