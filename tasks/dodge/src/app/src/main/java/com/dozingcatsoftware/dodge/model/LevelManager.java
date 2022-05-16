package com.dozingcatsoftware.dodge.model;

import androidx.annotation.NonNull;

import com.google.game.rl.RLTask;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * This class contains the logic for determining the number of types of bullets for each level.
 */
public class LevelManager {

    private static final int BASE_BULLETS = RLTask.get().get("base_bullets", 10);
    private static final int NEW_BULLETS_PER_LEVEL = RLTask.get().get("new_bullets_per_level", 3);

    private static final Class<? extends Bullet> DEFAULT_BULLET_CLASS = Bullet.class;

    // This structure defines the bullet frequencies as they change with levels
    private static final List<LevelConfig> LEVEL_INFO = Arrays.asList(
            new LevelConfig(5, Arrays.asList(
                    new LevelBulletConfig(90, Bullet.class),
                    new LevelBulletConfig(10, StopAndGoBullet.class)
            )),
            new LevelConfig(10, Arrays.asList(
                    new LevelBulletConfig(90, Bullet.class),
                    new LevelBulletConfig(5, StopAndGoBullet.class),
                    new LevelBulletConfig(5, SineWaveBullet.class)
            )),
            new LevelConfig(15, Arrays.asList(
                    new LevelBulletConfig(85, Bullet.class),
                    new LevelBulletConfig(10, StopAndGoBullet.class),
                    new LevelBulletConfig(5, SineWaveBullet.class)
            )),
            new LevelConfig(20, Arrays.asList(
                    new LevelBulletConfig(80, Bullet.class),
                    new LevelBulletConfig(10, StopAndGoBullet.class),
                    new LevelBulletConfig(10, SineWaveBullet.class)
            ))
    );

    private int currentLevel;
    private LevelConfig currentLevelConfig;

    public LevelManager() {
        setCurrentLevel(1);
    }

    public int getCurrentLevel() {
        return currentLevel;
    }

    public void setCurrentLevel(int level) {
        currentLevel = level;
        RLTask.get().logExtra("level [" + this.currentLevel + "]");

        LevelConfig newConfig = null;
        for (LevelConfig lc : LEVEL_INFO) {
            if (level >= lc.level) {
                newConfig = lc;
            } else {
                break;
            }
        }
        currentLevelConfig = newConfig;
    }

    private int numberOfBulletsForLevel(int level) {
        return BASE_BULLETS + (level - 1) * NEW_BULLETS_PER_LEVEL;
    }

    public int numberOfBulletsForCurrentLevel() {
        return numberOfBulletsForLevel(currentLevel);
    }

    @NonNull
    private Class<? extends Bullet> selectBulletClassForLevel(@SuppressWarnings("unused") int level) {
        if (currentLevelConfig == null) {
            return DEFAULT_BULLET_CLASS;
        }
        return currentLevelConfig.selectBulletClass();
    }

    @NonNull
    Class<? extends Bullet> selectBulletClassForCurrentLevel() {
        return selectBulletClassForLevel(this.currentLevel);
    }

    // Describes the bullet frequencies for a level. Contains a list of LevelBulletConfig objects which
    // have a bullet class and frequency weight.
    static class LevelConfig {
        final int level;
        @NonNull
        final List<LevelBulletConfig> bulletConfigs;
        final int totalFreq;
        static Random RAND = new Random();

        LevelConfig(int level, @NonNull List<LevelBulletConfig> bulletConfigs) {
            this.level = level;
            this.bulletConfigs = bulletConfigs;
            this.totalFreq = computeTotalFreq();
        }

        private int computeTotalFreq() {
            int totalFreq = 0;
            for (LevelBulletConfig bc : this.bulletConfigs) {
                totalFreq += bc.frequency;
            }
            return totalFreq;
        }

        @NonNull
        Class<? extends Bullet> bulletClassForValue(int value) {
            // keep subtracting frequencies from value until it hits 0
            // e.g. if frequencies are (10, 20, 70) 0-9 will pick the first, 10-29 second, rest third
            for (LevelBulletConfig bc : this.bulletConfigs) {
                value -= bc.frequency;
                if (value < 0) {
                    return bc.bulletClass;
                }
            }
            // shouldn't be here, take last
            return this.bulletConfigs.get(this.bulletConfigs.size() - 1).bulletClass;
        }

        @NonNull
        Class<? extends Bullet> selectBulletClass() {
            return bulletClassForValue(RAND.nextInt(this.totalFreq));
        }
    }

    static class LevelBulletConfig {
        final int frequency;
        @NonNull
        final Class<? extends Bullet> bulletClass;

        LevelBulletConfig(int frequency, @NonNull Class<? extends Bullet> bulletClass) {
            this.frequency = frequency;
            this.bulletClass = bulletClass;
        }
    }
}
