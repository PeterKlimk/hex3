mod app;

use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    env_logger::init();

    // Check for --stage2 flag to auto-advance to Stage 2 on startup
    let auto_stage2 = std::env::args().any(|arg| arg == "--stage2");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = app::App::new(auto_stage2);
    event_loop
        .run_app(&mut app)
        .expect("Failed to run application");
}
