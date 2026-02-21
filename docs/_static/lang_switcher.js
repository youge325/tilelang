/**
 * Language switcher for TileLang documentation.
 * Rewrites data-lang links so that clicking switches between /en/ and /zh_CN/
 * while preserving the current page path.
 */
document.addEventListener('DOMContentLoaded', function () {
  var switcher = document.getElementById('lang-switcher');
  if (!switcher) return;

  var links = switcher.querySelectorAll('a[data-lang]');
  links.forEach(function (link) {
    var targetLang = link.getAttribute('data-lang');
    var path = window.location.pathname;
    // Try to replace /en/ or /zh_CN/ segment in the URL
    var newPath = path.replace(/\/(en|zh_CN)\//, '/' + targetLang + '/');
    if (newPath === path) {
      // Fallback: just go to the root of the target language
      newPath = '../' + targetLang + '/index.html';
    }
    link.href = newPath;
  });
});
