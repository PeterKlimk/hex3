// Constants
const ITERATIONS: usize = 20;
const OVERSHOOT: f32 = 1.0; // Use 1.0 (Gauss-Seidel) for maximum stability on irregular grids
const EPSILON: f32 = 1e-6;  // For safe division

fn solve_consistent_atmosphere(tiles: &mut Vec<Tile>) {
    
    // --- STEP 1: CALCULATE DIVERGENCE (Net Flux) ---
    // Measures the volume of air entering/leaving the cell per frame.
    // Unit: [Area / Time] (in 2D)
    let mut divergence_cache = vec![0.0; tiles.len()];
    
    for (i, tile) in tiles.iter().enumerate() {
        let mut total_flux = 0.0;
        
        for (n_idx, &neighbor_id) in tile.neighbors.iter().enumerate() {
            let neighbor = &tiles[neighbor_id];
            let edge_len = tile.edge_lengths[n_idx];
            let permeability = tile.permeability[n_idx]; // 0.0 to 1.0
            
            // Normalized vector pointing to neighbor
            let normal = (neighbor.center - tile.center).normalize();

            // Flux = Velocity . Normal * Length * Permeability
            // Consistent Term A: (edge_len * permeability)
            let flux = tile.velocity.dot(normal) * (edge_len * permeability);
            
            total_flux += flux;
        }
        divergence_cache[i] = total_flux;
    }

    // --- STEP 2: SOLVE PRESSURE (Poisson Equation) ---
    // We solve for P such that Flux_Induced = -Divergence.
    // We assume Flux is driven by Pressure SLOPE (P/dist).
    for _iter in 0..ITERATIONS {
        for i in 0..tiles.len() {
            let tile = &tiles[i];
            
            let mut flux_induced = 0.0;
            let mut coeff_sum = 0.0;

            for (n_idx, &neighbor_id) in tile.neighbors.iter().enumerate() {
                let neighbor = &tiles[neighbor_id];
                
                let dist = (neighbor.center - tile.center).length().max(EPSILON);
                let edge_len = tile.edge_lengths[n_idx];
                let permeability = tile.permeability[n_idx];

                // Consistent Term B: (edge_len * permeability) / dist
                // This is the "Conductance" of the edge.
                let weight = (edge_len * permeability) / dist;

                flux_induced += (neighbor.pressure - tile.pressure) * weight;
                coeff_sum += weight;
            }

            if coeff_sum < EPSILON { continue; } // Isolated cell

            // SOR Step
            let residual = divergence_cache[i] + flux_induced;
            let change = -residual / coeff_sum;
            tiles[i].pressure += change * OVERSHOOT;
        }
    }

    // --- STEP 3: UPDATE VELOCITY (Subtract Gradient) ---
    // V_new = V_old - Grad(P)
    // We calculate Grad(P) using the Green-Gauss theorem, which sums
    // the flux of P across the boundary and divides by Area.
    for (i, tile) in tiles.iter_mut().enumerate() {
        let mut pressure_gradient_accum = Vec3::ZERO;

        for (n_idx, &neighbor_id) in tile.neighbors.iter().enumerate() {
            // Safe immutable lookups
            let neighbor_pressure = get_pressure_safe(tiles, neighbor_id);
            let neighbor_center = get_center_safe(tiles, neighbor_id);
            
            let dist = (neighbor_center - tile.center).length().max(EPSILON);
            let edge_len = tile.edge_lengths[n_idx];
            let permeability = tile.permeability[n_idx];
            let normal = (neighbor_center - tile.center).normalize();

            // Calculate the slope across this specific edge
            let pressure_slope = (neighbor_pressure - tile.pressure) / dist;

            // Consistent Term C: Match Step 2's assumption!
            // In Step 2, we said Flux ~ Slope * (Length * Perm).
            // Therefore, VelocityChange ~ Slope * (Length * Perm) / Area.
            
            // The contribution of this edge to the cell's total gradient vector:
            let grad_contribution = normal * (pressure_slope * edge_len * permeability);
            
            pressure_gradient_accum += grad_contribution;
        }

        // Divide by Area (Green-Gauss requirement)
        // This converts the sum of boundary forces into a volumetric gradient.
        let grad_p = pressure_gradient_accum / tile.area;

        // Apply
        tile.velocity -= grad_p;
        
        // --- HYDROLOGY OUTPUT ---
        // Uplift Logic remains the same:
        // High Pressure means the solver had to fight hard to stop flux.
        // Therefore, High Pressure = Strong Wind hitting Wall.
        let mechanical = tile.velocity.dot(tile.slope);
        let static_pressure = f32::max(0.0, tile.pressure);
        tile.uplift = mechanical + (static_pressure * 0.1);
    }
}