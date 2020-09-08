use super::VertexData;
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
pub struct ModelFace {
    pub vertex_0: usize,
    pub vertex_1: usize,
    pub vertex_2: usize,
}

pub struct Model {
    pub vertices: Vec<VertexData>,
    pub faces: Vec<ModelFace>,
}

impl Model {
    pub fn get_face_vertices(&self, face: &ModelFace) -> (&VertexData, &VertexData, &VertexData) {
        let v0 = &self.vertices[face.vertex_0];
        let v1 = &self.vertices[face.vertex_1];
        let v2 = &self.vertices[face.vertex_2];
        (v0, v1, v2)
    }
}
