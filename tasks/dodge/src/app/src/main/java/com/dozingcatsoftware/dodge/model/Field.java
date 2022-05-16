package com.dozingcatsoftware.dodge.model;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.google.game.rl.RLTask;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Represents the field on which the player and bullets move. The field's width is always normalized to 1.0,
 * and the height will vary based on the aspect ratio of the field's view. The Field object contains a Dodger
 * and list of Bullets, and updates their positions and checks for collisions in the tick() method.
 */
public class Field {

    /**
     * Interface with callback methods for when the dodger reaches the goal area or collides with a bullet.
     */
    public interface Delegate {
        void dodgerReachedGoal(Field field);

        void dodgerHitByBullet(Field field);
    }

    @Nullable
    private final Delegate delegate;
    @NonNull
    private final LevelManager levelManager;

    private double aspectRatio = 1.0d; // width is always 1, aspectRatio is height
    private static final double GOAL_RATIO = RLTask.get().get("goal_ratio", 0.075f); // fraction of field containing start and end areas
    private int maxBullets;

    @NonNull
    private final Random random = new Random();

    @Nullable
    private Bullet[] bullets;
    @Nullable
    private Dodger dodger;
    @SuppressWarnings("unused")
    private boolean running = false;
    private boolean movingUp = true;

    public Field(@Nullable Delegate delegate,
                 @NonNull LevelManager levelManager) {
        this.delegate = delegate;
        this.levelManager = levelManager;
    }

    public double goalHeight() {
        return GOAL_RATIO * aspectRatio;
    }

    /**
     * Creates and initializes a new bullet. The bullet type is determined by calling the LevelManager object,
     * and the speed and direction is randomly assigned. The bullet will always travel from one edge of the
     * screen to the opposite edge.
     */
    @NonNull
    private Bullet newBullet() {
        // LevelManager call returns a Class but we have to instantiate it here, sort of ugly
        Bullet bullet;
        Class<? extends Bullet> bulletClass = levelManager.selectBulletClassForCurrentLevel();
        if (bulletClass == SineWaveBullet.class) {
            bullet = SineWaveBullet.create(0.2 + 0.1 * random.nextDouble(), 0.1, 1.5);
        } else if (bulletClass == StopAndGoBullet.class) {
            bullet = StopAndGoBullet.create(0.2 + 0.2 * random.nextDouble(), 1.0, 1.0);
        } else if (bulletClass == HelixBullet.class) {
            bullet = HelixBullet.create(0.05 + 0.05 * random.nextDouble(), 0.1, 2);
        } else {
            // TODO: make speed configurable
            double speed = 0.2 + 0.2 * random.nextDouble();
            if (RLTask.get().isEnabled()) {
                float speedF = RLTask.get().get("speed", -1F);
                if (speedF > 0.0F) {
                    speed = speedF;
                }
            }
            bullet = Bullet.create(speed);
        }

        // set random position and direction, always going across the screen
        double p1 = random.nextDouble();
        double p2 = random.nextDouble();

        switch (random.nextInt(4)) {
            case 0: // bottom to top
                bullet.setPosition(new Vec2(p1, GOAL_RATIO * aspectRatio));
                bullet.setTargetPosition(new Vec2(p2, (1 - GOAL_RATIO) * aspectRatio));
                break;
            case 1: // top to bottom
                bullet.setPosition(new Vec2(p1, (1 - GOAL_RATIO) * aspectRatio));
                bullet.setTargetPosition(new Vec2(p2, GOAL_RATIO * aspectRatio));
                break;
            case 2: // left to right
                bullet.setPosition(new Vec2(0, (GOAL_RATIO + (1 - 2 * GOAL_RATIO) * p1) * aspectRatio));
                bullet.setTargetPosition(new Vec2(1, (GOAL_RATIO + (1 - 2 * GOAL_RATIO) * p2) * aspectRatio));
                break;
            default: // right to left
                bullet.setPosition(new Vec2(1, (GOAL_RATIO + (1 - 2 * GOAL_RATIO) * p1) * aspectRatio));
                bullet.setTargetPosition(new Vec2(0, (GOAL_RATIO + (1 - 2 * GOAL_RATIO) * p2) * aspectRatio));
        }

        bullet.setColor(new int[]{
                100 + random.nextInt(155),
                100 + random.nextInt(155),
                100 + random.nextInt(155)
        });
        return bullet;
    }

    private List<Integer> bulletIndexesToRemove = new ArrayList<>(); // avoid allocation in loop

    /**
     * Updates the position of the player and all bullets, handles collisions, and creates new bullets if
     * necessary.
     * Postcondition: this.bullets.length==maxBullets, and this.bullets has no null elements
     *
     * @param dt time interval since the last update
     */
    public void tick(double dt) {
        bulletIndexesToRemove.clear();

        if (this.bullets == null || this.bullets.length != maxBullets) {
            // allocate new array and copy existing bullets (this will truncate some bullets if current length>maxBullets)
            Bullet[] newBullets = new Bullet[maxBullets];
            if (this.bullets != null) {
                System.arraycopy(this.bullets, 0, newBullets, 0, Math.min(bullets.length, maxBullets));
            }
            this.bullets = newBullets;
        }

        for (int i = 0; i < maxBullets; i++) {
            Bullet b = this.bullets[i];
            if (b != null) {
                b.tick(dt);
                if (b.shouldRemoveFromField(this)) {
                    bulletIndexesToRemove.add(i);
                }
            }
        }

        // update dodger position and check for reaching goal or collision with bullet
        if (dodger != null) {
            dodger.tick(dt);
            Vec2 dPos = dodger.getPosition();
            if (!movingUp && dPos.y < goalHeight()
                    || movingUp && dPos.y > aspectRatio - goalHeight()) {
                // made it to goal
                movingUp = !movingUp;
                if (delegate != null) {
                    delegate.dodgerReachedGoal(this);
                }
            }
            // check for collision if player is outside of goal areas
            if (dPos.y > goalHeight()
                    && dPos.y < aspectRatio - goalHeight()) {
                boolean hit = false;
                for (int i = 0; i < maxBullets; i++) {
                    Bullet b = this.bullets[i];
                    if (b != null) {
                        if (dPos.squaredDistanceTo(b.getPosition()) < 0.025 * 0.025) {
                            hit = true;
                            bulletIndexesToRemove.add(i);
                        }
                    }
                }
                if (hit) {
                    if (delegate != null) {
                        delegate.dodgerHitByBullet(this);
                    }
                }
            }
        }

        // remove bullets at indexes to delete, and fill all null spots in bullets array
        for (int i = 0; i < bulletIndexesToRemove.size(); i++) {
            this.bullets[bulletIndexesToRemove.get(i)] = null;
        }

        for (int i = 0; i < maxBullets; i++) {
            if (this.bullets[i] == null) {
                this.bullets[i] = newBullet();
            }
        }
    }

    /**
     * Creates the player sprite and positions it in the center of the start area.
     */
    public void createDodger() {
        dodger = new Dodger();
        double yOffset = goalHeight() / 2;
        dodger.setPosition(new Vec2(0.5, movingUp ? yOffset : aspectRatio - yOffset));
        dodger.setSpeed(0.4);
        dodger.setLimits(0.02, yOffset, 0.98, aspectRatio - yOffset);
    }

    /**
     * Removes the player sprite.
     */
    public void removeDodger() {
        dodger = null;
    }

    public void start() {
        bullets = new Bullet[0];
        running = true;
    }

    @SuppressWarnings("unused")
    public void stop() {
        running = false;
    }

    // autogenerated

    public void setAspectRatio(double aspectRatio) {
        this.aspectRatio = aspectRatio;
    }

    public void setMaxBullets(int maxBullets) {
        this.maxBullets = maxBullets;
    }

    @Nullable
    public Bullet[] getBullets() {
        return bullets;
    }

    @Nullable
    public Dodger getDodger() {
        return dodger;
    }

    public boolean getMovingUp() {
        return movingUp;
    }
}
