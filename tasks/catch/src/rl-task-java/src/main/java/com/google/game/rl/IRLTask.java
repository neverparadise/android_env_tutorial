package com.google.game.rl;


@SuppressWarnings("unused")
public interface IRLTask {

  void logReady();
  @Deprecated
  void logExtra(Object extra);
  void logExtra(String name, Object value);
  void logScore(Object score);
  void logEpisodeEnd();

  IRLTask EMPTY_TASK = new IRLTask(){

    @Override
    public void logReady() {
      // DO NOTHING
    }

    @Deprecated
    @Override
    public void logExtra(Object extra) {
      // DO NOTHING
    }

    @Override
    public void logExtra(String name, Object value) {
      // DO NOTHING
    }

    @Override
    public void logScore(Object score) {
      // DO NOTHING
  }

    @Override
    public void logEpisodeEnd() {
      // DO NOTHING
    }
  };
}
