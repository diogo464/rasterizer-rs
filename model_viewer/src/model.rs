use rasterizer::math_prelude::*;

#[derive(Debug)]
pub struct ModelVertex {
    pub position: Vec3,
    pub texture: Vec2,
    pub normal: Vec3,
    pub tangent: Vec3,
    pub bitangent: Vec3,
}

#[derive(Debug)]
#[allow(dead_code)]
struct ModelFace {
    pub vertex_0: u32,
    pub vertex_1: u32,
    pub vertex_2: u32,
}

pub struct Model {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,
}
