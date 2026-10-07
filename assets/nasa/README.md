# NASA 3D models

Official models from NASA's 3D Resources collection (https://github.com/nasa/NASA-3D-Resources, https://nasa3d.arc.nasa.gov).
NASA states these assets are "free and without copyright"; use follows NASA's media usage guidelines (https://www.nasa.gov/nasa-brand-center/images-and-media).
No NASA endorsement of this project is implied.

| file | NASA model |
|---|---|
| astronaut.glb | Astronaut |
| apollo_lunar_module.glb | Apollo Lunar Module |
| rassor.glb | Regolith Advanced Surface Systems Operations Robot (RASSOR), simplified with gltf-transform 4.5.1 (`weld`, `simplify --ratio 0.1 --error 0.002`, `draco`) from 2.03 M to 0.20 M triangles and from 6.3 MB to 0.74 MB, so the 3-D view loads quickly. Lossy, and visually equivalent at the scale shown (0.4% of pixels differ in a close-up render) |
| perseverance.glb | Mars 2020 Perseverance Rover |
| gateway.glb | Gateway ("Gateway Core.glb", 66 MB) re-encoded with gltf-transform 4.5.1 (Draco geometry, WebP textures at 1024 px) to 3.8 MB for the web. Both steps are lossy (Draco quantises vertex positions; WebP is lossy), visually equivalent at web viewing scale |
| esas_crew_module.glb | ESAS Crew Module (design predecessor of Orion) |
