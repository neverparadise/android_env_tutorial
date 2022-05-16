package com.dozingcatsoftware.dodge.model;

/**
 * Bullet that increases speed over time. Not currently used.
 */
@SuppressWarnings("unused")
public class AcceleratingBullet extends Bullet {

    private double acceleration;

    public static AcceleratingBullet create(double speed, double accel) {
        AcceleratingBullet self = new AcceleratingBullet();
        self.setSpeed(speed);
        self.setAcceleration(accel);
        return self;
    }

    public void tick(double dt) {
        speed += acceleration * dt;
        super.tick(dt);
    }

    ////////////////////////////////////////////////////

    public double getAcceleration() {
        return acceleration;
    }

    private void setAcceleration(double acceleration) {
        this.acceleration = acceleration;
    }
}
