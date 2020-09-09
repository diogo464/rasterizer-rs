use glm::{Vec2, Vec3};
use nalgebra_glm as glm;

#[derive(Debug)]
pub struct ModelVertex {
    pub position: Vec3,
    pub texture: Option<Vec2>,
    pub normal: Option<Vec3>,
}

impl ModelVertex {
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

#[derive(Debug)]
#[allow(dead_code)]
struct ModelFace {
    pub vertex_0: usize,
    pub vertex_1: usize,
    pub vertex_2: usize,
}

pub struct Model {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<usize>,
}
