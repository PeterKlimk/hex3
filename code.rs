// Constants
const ITERATIONS: usize = 20;
const OVERSHOOT: f32 = 1.5; // Successive Over-Relaxation factor (speeds up convergence)
const AIR_DENSITY: f32 = 1.0; // Can be 1.0, just implies mass = area * density

struct Tile {
    // Inputs
    center: Vec3,
    neighbors: Vec<usize>,
    edge_lengths: Vec<f32>, // Length of the border with neighbor
    area: f32,              // CRITICAL: The "Mass" of the cell
    height: f32,

    // State
    velocity: Vec3,         // Starts as Zonal + Thermal
    pressure: f32,          // Starts at 0.0

    // Output
    rainfall_intensity: f32, 
}

fn solve_atmosphere(tiles: &mut Vec<Tile>) {
    
    // --- PRE-CALCULATION: Terrain Permeability ---
    // We determine which edges are "open" and which are "walls".
    // This must be consistent across all steps (Divergence, Solver, Gradient).
    let mut edge_weights: Vec<Vec<f32>> = calculate_permeability_weights(tiles);

    // --- STEP 1: CALCULATE DIVERGENCE (The Source) ---
    // We measure how much air is currently flowing in/out based on the *current* velocity.
    // Note: We do NOT divide by area yet. We want total "Flux" (Volume per second).
    let mut divergence_cache = vec![0.0; tiles.len()];
    
    for (i, tile) in tiles.iter().enumerate() {
        let mut total_flux = 0.0;
        
        for (n_idx, &neighbor_id) in tile.neighbors.iter().enumerate() {
            let neighbor = &tiles[neighbor_id];
            let edge_len = tile.edge_lengths[n_idx];
            let permeability = edge_weights[i][n_idx]; // 0.05 = wall, 1.0 = open

            // Calculate Flow Direction (Normal)
            // Ideally use pre-computed normals, but this works for Voronoi graphs
            let normal = (neighbor.center - tile.center).normalize();

            // Flux = (Speed in direction of neighbor) * (Width of gate) * (Openness of gate)
            let velocity_component = tile.velocity.dot(normal);
            let flux = velocity_component * edge_len * permeability;

            total_flux += flux;
        }
        divergence_cache[i] = total_flux;
    }

    // --- STEP 2: SOLVE PRESSURE (The Balance) ---
    // We try to find a Pressure field that pushes back exactly hard enough 
    // to cancel out that divergence.
    for _iter in 0..ITERATIONS {
        for i in 0..tiles.len() {
            let tile = &tiles[i];
            
            let mut flow_induced_by_neighbors = 0.0;
            let mut total_coefficient = 0.0;

            for (n_idx, &neighbor_id) in tile.neighbors.iter().enumerate() {
                let neighbor = &tiles[neighbor_id];
                let dist = (neighbor.center - tile.center).length();
                let edge_len = tile.edge_lengths[n_idx];
                let permeability = edge_weights[i][n_idx];

                // The Geometric Weight for the Laplacian
                // "How wide is the connection vs how long is the tunnel?"
                let geom_weight = edge_len / dist;
                
                // The Combined Weight
                // If permeability is low (wall), the solver knows it can't 
                // fix pressure easily through this edge.
                let weight = geom_weight * permeability;

                flow_induced_by_neighbors += (neighbor.pressure - tile.pressure) * weight;
                total_coefficient += weight;
            }

            // prevent div/0 for isolated cells (though rare in voronoi)
            if total_coefficient < 0.00001 { continue; }

            // The Relaxation Step
            // We want (flow_induced + divergence) to equal 0.
            let target_change = -(divergence_cache[i] + flow_induced_by_neighbors) / total_coefficient;
            
            // Apply with Over-Relaxation (OVERSHOOT) for speed
            tiles[i].pressure += target_change * OVERSHOOT;
        }
    }

    // --- STEP 3: APPLY FORCES (The Physics Fix) ---
    // Here we strictly apply F = ma.
    // This fixes the "Operator Mismatch" your agent found.
    for (i, tile) in tiles.iter_mut().enumerate() {
        let mut force_accum = Vec3::ZERO;

        for (n_idx, &neighbor_id) in tile.neighbors.iter().enumerate() {
            // Need immutable access to neighbor pressure, but we are in a mutable iter.
            // (Assuming you handle Rust borrow checker here via indexing or unsafe/splitting)
            let neighbor_pressure = get_pressure_safe(tiles, neighbor_id); 
            
            let edge_len = tile.edge_lengths[n_idx];
            let permeability = edge_weights[i][n_idx];
            let normal = (get_center_safe(tiles, neighbor_id) - tile.center).normalize();

            // CALCULATE FORCE
            // Pressure acts on the Wall Surface (Edge Length)
            // P = Force / Length  ->  Force = P * Length
            let pressure_diff = tile.pressure - neighbor_pressure;
            let force_magnitude = pressure_diff * edge_len * permeability;

            // Force vector pushes away from high pressure
            force_accum += normal * force_magnitude;
        }

        // CALCULATE ACCELERATION
        // a = F / m
        // On a 2D map, Mass is proportional to Area
        // This division is what fixes the "1000x scale mismatch"
        let acceleration = force_accum / (tile.area * AIR_DENSITY);

        // Update Velocity (Project out the divergence)
        tile.velocity -= acceleration;
    }

    // --- STEP 4: HYDROLOGY (The Payoff) ---
    for tile in tiles.iter_mut() {
        // A. Mechanical Lift (Standard)
        // Wind physically climbing a slope
        let lift_mechanical = tile.velocity.dot(tile.slope_vector);

        // B. Pressure Lift (The "Counterfactual")
        // The solver spiked the pressure here to stop wind from going through a wall.
        // That spike represents the "blocked momentum" forcing air upwards.
        // We clamp to 0 because negative pressure (suction) is dry.
        let lift_pressure = f32::max(0.0, tile.pressure);
        
        // C. Combine
        // '0.1' is a tuning factor to balance pressure units vs velocity units
        tile.rainfall_intensity = f32::max(0.0, lift_mechanical + (lift_pressure * 0.1));
    }
}