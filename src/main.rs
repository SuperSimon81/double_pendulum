use nannou::prelude::*;
use ndarray::{array, Array1};
use std::vec;
use std::{f32, f64};
use rayon::prelude::*;
use std::time::{Instant};

//Length of pendulum
const L: f64 = 100.0;
const PI: f64 = 3.14159265359;
const TIME_STEP: f64 = 0.10159265359;
//Offset to starting condition

const NUM_PENDULUMS: u32 = 10000;
const OFFSET : f64 = 0.1/NUM_PENDULUMS as f64;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    _window: window::Id,
    list: Vec<Array1<f64>>,
    calc_time: u128,
}

fn model(app: &App) -> Model {
    //Make a window, set size etc..
    let _window = app.new_window().size(500, 500).view(view).build().unwrap();
    let calc_time = 0;
    //let draw_time = 0;

    //Mutable vector of arrays for all the pendulums, 
    //th1,th2,th1_v and th2_v is the angle 
    //of the first and second arm and the 
    //velocity of the first and second arm
    let mut list: Vec<Array1<f64>> = Vec::new(); 
    for n in 0..NUM_PENDULUMS {
        list.push(array![PI-n as f64 * OFFSET/3.0+0.2,PI+n as f64 * OFFSET-0.2,   0.0, 0.0]);
    }

    Model {
        _window,
        list,
        calc_time,
    }
}

fn eqs(y: Array1<f64>) -> Array1<f64> {
    let l1 = L;
    let l2 = L;

    let m1 = 1.0;
    let m2 = 1.0;

    let g = 9.8;
    let th1 = y[0];
    let th2 = y[1];
    let th1_v = y[2];
    let th2_v = y[3];

    let denominator: f64 = 2.0 * m1 + m2 - m2 * (2.0 * th1 - 2.0 * th2).cos();
    let th1_a: f64 = (-g * (2.0 * m1 + m2) * th1.sin()
        - m2 * g * (th1 - 2.0 * th2).sin()
        - 2.0
            * (th1 - th2).sin()
            * m2
            * (th2_v.powi(2) * l2 + th1_v.powi(2) * l1 * (th1 - th2).cos()))
        / (l1 * denominator);
    let th2_a: f64 = 2.0
        * (th1 - th2).sin()

        * (th1_v.powi(2) * l1 * (m1 + m2)
            + g * (m1 + m2) * th1.cos()
            + th2_v.powi(2) * l2 * m2 * (th1 - th2).cos())
        / (l2 * denominator);

    let arr: Array1<f64> = array![th1_v, th2_v, th1_a, th2_a];
    arr
}

fn update(_app: &App, _model: &mut Model, _update: Update) {
    
    let now = Instant::now();
    //Fun way to use parallell iter, which doesnt improve runtime because
    //rendering is the slow part. TODO wgpu vertexbuffer rendering
    _model.list.par_iter_mut().for_each(|p| *p = p.clone() + runge_kutta4(&eqs, p.clone(), TIME_STEP));
    _model.calc_time = now.elapsed().as_millis();
}

fn runge_kutta4(fx: &dyn Fn(Array1<f64>) -> Array1<f64>, y: Array1<f64>, dx: f64) -> Array1<f64> {
    let k1: Array1<f64> = dx * fx(y.clone());
    let k2: Array1<f64> = dx * fx(y.clone() + k1.clone() / 2.0);
    let k3: Array1<f64> = dx * fx(y.clone() + k2.clone() / 2.0);
    let k4: Array1<f64> = dx * fx(y + k3.clone());

    (k1.clone() + 2.0 * k2.clone() + 2.0 * k3.clone() + k4.clone()) / 6.0
}

fn view(app: &App, _model: &Model, frame: Frame) {
   
    let draw = app.draw();
    
    draw.background().color(BLACK);
    let mut count = 0.0;
    
    for pendulum in _model.list.iter() {
        //Have some fun with color
        let c  = hsva(count/NUM_PENDULUMS as f32,1.0,1.0,0.01);
        count +=1.0;
        let x = L as f32 * (pendulum[0] - PI / 2.0).cos() as f32;
        let y = L as f32 * (pendulum[0] - PI / 2.0).sin() as f32;
        let x2 = x + L as f32 * (pendulum[1] - PI / 2.0).cos() as f32;
        let y2 = y + L as f32 * (pendulum[1] - PI / 2.0).sin() as f32;

        let offset = pt2(0.0, 0.0);

        draw.line()
            .start(pt2(0.0, 0.0) - offset)
            .end(pt2(x as f32, y as f32) - offset)
            .weight(1.0)
            .color(c);
        draw.line()
            .start(pt2(x as f32, y as f32) - offset)
            .end(pt2(x2 as f32, y2 as f32) - offset)
            .weight(1.0)
            .color(c);
        
        //If you want to have balls attached to the last arm
        /*draw.ellipse()
            .color(c)
            .xy(pt2(x as f32, y as f32) - offset)
            .width(10.0)
            .height(10.0);*/
        /*draw.ellipse()
            .color(c)
            .xy(pt2(x2 as f32, y2 as f32) - offset)
            .width(10.0)
            .height(10.0);*/
    }
    

    //Some debug text for fun
    //draw.text(&format!("elapsed {}, since last update {}, calc {}",app.elapsed_frames(),app.duration.since_prev_update.as_millis(),_model.calc_time))
    //.x_y(-150.0,230.0)
    //.rgb(1., 1., 1.);
    draw.to_frame(app, &frame).unwrap();
    
    //Do you want to capture to disk in order to make a gif?
    //let file_path = capture_frame_to_path(app, &frame);
    //app.main_window().capture_frame(file_path);

    //println!("{}",app.elapsed_frames());
    if app.elapsed_frames()>1023
    {
        app.quit();
    }
}


//If you would like to render to disk

/*fn capture_frame_to_path(app: &App, frame: &Frame) -> std::path::PathBuf {
app.project_path()
    .expect("failed to locate `project_path`")
    .join(app.exe_name().unwrap())
    .join(format!("{:03}", frame.nth()))
    .with_extension("png")
}*/