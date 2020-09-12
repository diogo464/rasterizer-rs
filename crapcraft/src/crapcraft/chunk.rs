use super::block::Block;
use glm::Vec3;
use nalgebra_glm as glm;

const fn chunk_block_count(size: usize) -> usize {
    size * size * size
}

pub struct ChunkVertex {
    pub position: Vec3,
}

impl ChunkVertex {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: Vec3::new(x, y, z),
        }
    }
}

pub struct ChunkMesh {
    pub vertices: Vec<ChunkVertex>,
    pub indices: Vec<usize>,
}

impl ChunkMesh {
    pub fn new(vertices: Vec<ChunkVertex>, indices: Vec<usize>) -> Self {
        Self { vertices, indices }
    }
}

impl Default for ChunkMesh {
    fn default() -> Self {
        Self::new(Vec::new(), Vec::new())
    }
}

pub struct Chunk {
    //x z y
    blocks: [Block; chunk_block_count(Self::SIZE)],
}

impl Chunk {
    const SIZE: usize = 16;

    pub fn new() -> Self {
        Self::filled(Block::Air)
    }

    pub fn filled(block: Block) -> Self {
        Self {
            blocks: [block; chunk_block_count(Self::SIZE)],
        }
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn blocks_mut(&mut self) -> &mut [Block] {
        &mut self.blocks
    }

    pub fn generate_mesh(&self) -> ChunkMesh {
        enum Dir {
            Up,
            Down,
            Left,
            Right,
            Front,
            Back,
        }

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let mut add_face = |x, y, z, dir| {
            let x = x as f32;
            let y = y as f32;
            let z = z as f32;
            let indices_start = vertices.len();
            match dir {
                Dir::Up => {
                    vertices.push(ChunkVertex::new(x, y + 1.0, z));
                    vertices.push(ChunkVertex::new(x + 1.0, y + 1.0, z));
                    vertices.push(ChunkVertex::new(x, y + 1.0, z + 1.0));
                    vertices.push(ChunkVertex::new(x + 1.0, y + 1.0, z));
                    vertices.push(ChunkVertex::new(x + 1.0, y + 1.0, z + 1.0));
                    vertices.push(ChunkVertex::new(x, y + 1.0, z + 1.0));
                }
                Dir::Down => {
                    vertices.push(ChunkVertex::new(x, y, z));
                    vertices.push(ChunkVertex::new(x + 1.0, y, z));
                    vertices.push(ChunkVertex::new(x + 1.0, y, z + 1.0));
                    vertices.push(ChunkVertex::new(x, y, z));
                    vertices.push(ChunkVertex::new(x + 1.0, y, z + 1.0));
                    vertices.push(ChunkVertex::new(x, y, z + 1.0));
                }
                Dir::Left => {
                    vertices.push(ChunkVertex::new(x, y, z));
                    vertices.push(ChunkVertex::new(x, y + 1.0, z));
                    vertices.push(ChunkVertex::new(x, y + 1.0, z + 1.0));
                    vertices.push(ChunkVertex::new(x, y, z));
                    vertices.push(ChunkVertex::new(x, y, z + 1.0));
                    vertices.push(ChunkVertex::new(x, y + 1.0, z + 1.0));
                }
                Dir::Right => {
                    vertices.push(ChunkVertex::new(x + 1.0, y, z));
                    vertices.push(ChunkVertex::new(x + 1.0, y + 1.0, z));
                    vertices.push(ChunkVertex::new(x + 1.0, y + 1.0, z + 1.0));
                    vertices.push(ChunkVertex::new(x + 1.0, y, z));
                    vertices.push(ChunkVertex::new(x + 1.0, y, z + 1.0));
                    vertices.push(ChunkVertex::new(x + 1.0, y + 1.0, z + 1.0));
                }
                Dir::Front => {
                    vertices.push(ChunkVertex::new(x, y, z));
                    vertices.push(ChunkVertex::new(x + 1.0, y, z));
                    vertices.push(ChunkVertex::new(x + 1.0, y + 1.0, z));
                    vertices.push(ChunkVertex::new(x, y, z));
                    vertices.push(ChunkVertex::new(x, y + 1.0, z));
                    vertices.push(ChunkVertex::new(x + 1.0, y + 1.0, z));
                }
                Dir::Back => {
                    vertices.push(ChunkVertex::new(x, y, z + 1.0));
                    vertices.push(ChunkVertex::new(x + 1.0, y, z + 1.0));
                    vertices.push(ChunkVertex::new(x + 1.0, y + 1.0, z + 1.0));
                    vertices.push(ChunkVertex::new(x, y, z + 1.0));
                    vertices.push(ChunkVertex::new(x, y + 1.0, z + 1.0));
                    vertices.push(ChunkVertex::new(x + 1.0, y + 1.0, z + 1.0));
                }
            }
            (0..6).for_each(|i| indices.push(indices_start + i))
        };

        let block_missing_or_air = |x, y, z| {
            if let Some(b) = self.get_block_at_pos(x, y, z) {
                b == Block::Air
            } else {
                true
            }
        };

        for (index, b) in self.blocks.iter().enumerate() {
            if *b == Block::Air {
                continue;
            }
            let (x, y, z) = Self::index_to_coords(index);
            //up
            if block_missing_or_air(x, y + 1, z) {
                add_face(x, y, z, Dir::Up);
            }
            //down
            if block_missing_or_air(x, y - 1, z) {
                add_face(x, y, z, Dir::Down);
            }
            //Left
            if block_missing_or_air(x - 1, y, z) {
                add_face(x, y, z, Dir::Left);
            }
            //Right
            if block_missing_or_air(x + 1, y, z) {
                add_face(x, y, z, Dir::Right);
            }
            //Front
            if block_missing_or_air(x, y, z - 1) {
                add_face(x, y, z, Dir::Front);
            }
            //Back
            if block_missing_or_air(x, y, z + 1) {
                add_face(x, y, z, Dir::Back);
            }
        }

        ChunkMesh::new(vertices, indices)
    }

    fn index_to_coords(index: usize) -> (usize, usize, usize) {
        chunk_index_to_coords(Self::SIZE, index)
    }

    fn coords_to_index(x: usize, y: usize, z: usize) -> Option<usize> {
        chunk_coords_to_index(Self::SIZE, x, y, z)
    }

    fn get_block_at_pos(&self, x: usize, y: usize, z: usize) -> Option<Block> {
        Self::coords_to_index(x, y, z).map(|i| self.blocks.get(i).map(|b| *b).unwrap())
    }
}

#[inline]
fn chunk_index_to_coords(chunk_size: usize, index: usize) -> (usize, usize, usize) {
    let x = index % chunk_size;
    let y = index / (chunk_size * chunk_size);
    let z = (index % (chunk_size * chunk_size)) / chunk_size;
    (x, y, z)
}

#[inline]
fn chunk_coords_to_index(chunk_size: usize, x: usize, y: usize, z: usize) -> Option<usize> {
    if x >= chunk_size || y >= chunk_size || z >= chunk_size {
        None
    } else {
        Some(x + y * chunk_size * chunk_size + z * chunk_size)
    }
}

#[test]
fn test_index_to_coords() {
    assert_eq!(chunk_index_to_coords(2, 2), (0, 0, 1));
    assert_eq!(chunk_index_to_coords(2, 5), (1, 1, 0));
}
