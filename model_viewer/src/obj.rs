use std::path::Path;

use rasterizer::math_prelude::*;

use crate::model::{Model, ModelVertex};

struct ObjModel {
    vertices: Vec<Vec3>,
    normals: Vec<Vec3>,
    texture: Vec<Vec2>,
    faces: Vec<ObjFace>,
}

struct ObjFace {
    vertices: [usize; 3],
    normals: Option<[usize; 3]>,
    textures: Option<[usize; 3]>,
}

fn read_obj_model<P: AsRef<Path>>(path: P) -> ObjModel {
    let contents = std::fs::read_to_string(path).unwrap();

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut texture = Vec::new();
    let mut faces = Vec::new();

    for line in contents.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        let line = line.replace("  ", " ");
        let mut split = line.split(' ');
        //println!("{}", line);
        match split.next().unwrap() {
            "v" => {
                let x = split.next().unwrap().parse::<f32>().unwrap();
                let y = split.next().unwrap().parse::<f32>().unwrap();
                let z = split.next().unwrap().parse::<f32>().unwrap();
                vertices.push(Vec3::new(x, y, z));
            }
            "vn" => {
                let x = split.next().unwrap().parse::<f32>().unwrap();
                let y = split.next().unwrap().parse::<f32>().unwrap();
                let z = split.next().unwrap().parse::<f32>().unwrap();
                normals.push(Vec3::new(x, y, z));
            }
            "vt" => {
                let x = split.next().unwrap().parse::<f32>().unwrap();
                let y = split.next().unwrap().parse::<f32>().unwrap();
                texture.push(Vec2::new(x, y));
            }
            "f" => {
                let mut x_part = split.next().unwrap().split('/');
                let mut y_part = split.next().unwrap().split('/');
                let mut z_part = split.next().unwrap().split('/');

                let v0 = x_part.next().unwrap().parse::<usize>().unwrap() - 1;
                let v1 = y_part.next().unwrap().parse::<usize>().unwrap() - 1;
                let v2 = z_part.next().unwrap().parse::<usize>().unwrap() - 1;

                let textures = if let Some(n0) = x_part.nth(0) {
                    let t0 = n0.parse::<usize>().unwrap();
                    let t1 = y_part.nth(0).unwrap().parse::<usize>().unwrap();
                    let t2 = z_part.nth(0).unwrap().parse::<usize>().unwrap();
                    Some([t0 - 1, t1 - 1, t2 - 1])
                } else {
                    None
                };

                let normals = if let Some(n0) = x_part.nth(0) {
                    let n0 = n0.parse::<usize>().unwrap();
                    let n1 = y_part.nth(0).unwrap().parse::<usize>().unwrap();
                    let n2 = z_part.nth(0).unwrap().parse::<usize>().unwrap();
                    Some([n0 - 1, n1 - 1, n2 - 1])
                } else {
                    None
                };

                faces.push(ObjFace {
                    vertices: [v0, v1, v2],
                    textures,
                    normals,
                })
            }
            _ => continue,
        }
    }

    ObjModel {
        vertices,
        texture,
        normals,
        faces,
    }
}

pub fn read_model<P: AsRef<Path>>(path: P) -> Model {
    use std::convert::TryFrom;

    let obj = read_obj_model(path);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for face in obj.faces.iter() {
        let v0_pos = obj.vertices[face.vertices[0]];
        let v1_pos = obj.vertices[face.vertices[1]];
        let v2_pos = obj.vertices[face.vertices[2]];

        let v0_normal = face
            .normals
            .map(|indices| obj.normals[indices[0]])
            .unwrap_or_default();
        let v1_normal = face
            .normals
            .map(|indices| obj.normals[indices[1]])
            .unwrap_or_default();
        let v2_normal = face
            .normals
            .map(|indices| obj.normals[indices[2]])
            .unwrap_or_default();

        let v0_texture = face
            .textures
            .map(|indices| obj.texture[indices[0]])
            .unwrap_or_default();
        let v1_texture = face
            .textures
            .map(|indices| obj.texture[indices[1]])
            .unwrap_or_default();
        let v2_texture = face
            .textures
            .map(|indices| obj.texture[indices[2]])
            .unwrap_or_default();

        let v0_index = vertices.len();
        let v1_index = v0_index + 1;
        let v2_index = v0_index + 2;

        let (tangent, bitangent) = calc_tangent_bitangent(
            &[v0_pos, v1_pos, v2_pos],
            &[v0_texture, v1_texture, v2_texture],
        );
        vertices.push(ModelVertex {
            position: v0_pos,
            normal: v0_normal,
            texture: v0_texture,
            tangent,
            bitangent,
        });

        vertices.push(ModelVertex {
            position: v1_pos,
            normal: v1_normal,
            texture: v1_texture,
            tangent,
            bitangent,
        });

        vertices.push(ModelVertex {
            position: v2_pos,
            normal: v2_normal,
            texture: v2_texture,
            tangent,
            bitangent,
        });

        indices.push(v0_index);
        indices.push(v1_index);
        indices.push(v2_index);
    }

    Model {
        vertices,
        indices: indices
            .into_iter()
            .map(|i| u32::try_from(i).unwrap())
            .collect(),
    }
}

pub fn calc_tangent_bitangent(vs: &[Vec3; 3], uvs: &[Vec2; 3]) -> (Vec3, Vec3) {
    let delta_u_v1 = uvs[1] - uvs[0];
    let delta_u_v2 = uvs[2] - uvs[0];
    let edge1 = vs[1] - vs[0];
    let edge2 = vs[2] - vs[0];

    let f = 1.0 / (delta_u_v1.x * delta_u_v2.y - delta_u_v2.x * delta_u_v1.y);

    let mut tangent1 = Vec3::default();
    let mut bitangent1 = Vec3::default();
    tangent1.x = f * (delta_u_v2.y * edge1.x - delta_u_v1.y * edge2.x);
    tangent1.y = f * (delta_u_v2.y * edge1.y - delta_u_v1.y * edge2.y);
    tangent1.z = f * (delta_u_v2.y * edge1.z - delta_u_v1.y * edge2.z);

    bitangent1.x = f * (-delta_u_v2.x * edge1.x + delta_u_v1.x * edge2.x);
    bitangent1.y = f * (-delta_u_v2.x * edge1.y + delta_u_v1.x * edge2.y);
    bitangent1.z = f * (-delta_u_v2.x * edge1.z + delta_u_v1.x * edge2.z);

    (tangent1, bitangent1)

    //let e1 = vs[1] - vs[0];
    //let e2 = vs[2] - vs[0];
    //let duv1 = uvs[1] - uvs[0];
    //let duv2 = uvs[2] - uvs[0];

    //let inv = Mat2::from_cols(duv1, duv2).inverse();
    //let r1 = inv.row(0);
    //let r2 = inv.row(1);
    //let c1 = Vec2::new(e1.x, e2.x);
    //let c2 = Vec2::new(e1.y, e2.y);
    //let c3 = Vec2::new(e1.z, e2.z);
    //let tangent = Vec3::new(r1.dot(c1), r1.dot(c2), r1.dot(c3));
    //let bitangent = Vec3::new(r2.dot(c1), r2.dot(c2), r2.dot(c3));

    //(tangent, bitangent)
}
