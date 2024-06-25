use glam::{Mat4, Vec3};

use crate::{
    model::Model,
    obj,
    shader::{PBRFragmentShader, PBRUniform, PBRVertexShader},
    texture::Texture,
};

use super::{ModelViewer, Scene};

pub struct PBRScene {
    model: Model,
    uniform: PBRUniform,
    vertex: PBRVertexShader,
    fragment: PBRFragmentShader,
}

impl PBRScene {
    pub fn new(
        model: &str,
        albedo: &str,
        normal: &str,
        ao: &str,
        rough: &str,
        metal: &str,
    ) -> Self {
        let model = obj::read_model(model);
        let uniform = PBRUniform {
            model: Default::default(),
            view: Default::default(),
            projection: Default::default(),
            viewpos: Default::default(),
            light_pos: Default::default(),
            albedo_tex: Texture::load(albedo),
            normal_tex: Texture::load(normal),
            ao_tex: Texture::load(ao),
            roughness_tex: Texture::load(rough),
            metallic_tex: Texture::load(metal),
        };

        Self {
            model,
            uniform,
            vertex: PBRVertexShader,
            fragment: PBRFragmentShader,
        }
    }

    pub fn load() -> Self {
        Self::new(
            "models/cerberus/cerberus_mesh.obj",
            "models/cerberus/cerberus_albedo.png",
            "models/cerberus/cerberus_normal.png",
            "models/cerberus/cerberus_ao.png",
            "models/cerberus/cerberus_rough.png",
            "models/cerberus/cerberus_metal.png",
        )
    }
}

impl Scene for PBRScene {
    fn render(&mut self, model_viewer: &mut ModelViewer) {
        self.uniform.view = model_viewer.camera.generate_view_matrix();
        self.uniform.model = Mat4::IDENTITY;
        self.uniform.projection = model_viewer.camera.generate_projection_matrix();
        self.uniform.viewpos = model_viewer.camera.position;
        self.uniform.light_pos = Vec3::new(0.3, 1.2, 0.4);

        model_viewer.rasterizer.render_mesh(
            &self.model.vertices,
            &self.model.indices,
            &self.vertex,
            &self.fragment,
            &self.uniform,
        );
    }
}
