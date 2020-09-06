use super::VertexData;

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
