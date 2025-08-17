/**
 * Add listener for theme mode toggle
 */
const $toggle = document.getElementById('mode-toggle');
export function modeWatcher() {
  console.log('Mode watcher initialized');
  if (!$toggle) {
    return;
  }

  $toggle.addEventListener('click', () => {
    Theme.flip();
  });
}
