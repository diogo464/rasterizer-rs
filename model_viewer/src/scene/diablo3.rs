use glam::{Mat4, Vec3, Vec4};

use crate::{
    model::Model,
    obj,
    shader::{D3FragmentShader, D3Uniform, D3VertexShader},
    texture::Texture,
};

use super::{ModelViewer, Scene};

pub struct DiabloScene {
    model: Model,
    vertex: D3VertexShader,
    fragment: D3FragmentShader,
    uniform: D3Uniform,
}

impl DiabloScene {
    pub fn new(model: &str, diffuse: &str, normal: &str, glow: &str, specular: &str) -> Self {
        let model = obj::read_model(model);
        let var_name = D3Uniform {
            pvm: Default::default(),
            light_position: Default::default(),
            light_color: Vec3::ONE,
            light_intensity: 0.7,
            ambient_color: Vec3::ONE,
            ambient_intensity: 0.3,
            view_pos: Default::default(),
            diffuse: Texture::load(diffuse),
            specular: Texture::load(specular),
            normals: Texture::load(normal),
            glow: Texture::load(glow),
        };
        let uniform = var_name;

        Self {
            model,
            uniform,
            vertex: D3VertexShader,
            fragment: D3FragmentShader,
        }
    }

    pub fn load() -> Self {
        let model = "models/diablo3/diablo3_pose.obj";
        let diffuse = "models/diablo3/diablo3_pose_diffuse.tga";
        let normal = "models/diablo3/diablo3_pose_nm.tga";
        let glow = "models/diablo3/diablo3_pose_glow.tga";
        let _tangent = "models/diablo3/diablo3_pose_nm_tangent.tga";
        let specular = "models/diablo3/diablo3_pose_spec.tga";
        Self::new(model, diffuse, normal, glow, specular)
    }
}

impl Scene for DiabloScene {
    fn render(&mut self, model_viewer: &mut ModelViewer) {
        self.uniform.pvm.view = model_viewer.camera.generate_view_matrix();
        self.uniform.pvm.model = Mat4::IDENTITY;
        self.uniform.pvm.projection = model_viewer.camera.generate_projection_matrix();
        self.uniform.view_pos = model_viewer.camera.position;
        self.uniform.light_position = model_viewer.light_position;

        model_viewer.rasterizer.render_mesh(
            &self.model.vertices,
            &self.model.indices,
            &self.vertex,
            &self.fragment,
            &self.uniform,
        );
    }
}
