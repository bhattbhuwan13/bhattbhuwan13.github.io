document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.highlighter-rouge').forEach(function (block) {
    var codeEl = block.querySelector('pre.highlight code') || block.querySelector('pre.highlight');
    if (!codeEl) return;

    var button = document.createElement('button');
    button.className = 'copy-code-button';
    button.type = 'button';
    button.setAttribute('aria-label', 'Copy code to clipboard');
    var defaultHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M4 2h7v2H6v9H4V2z" fill="currentColor"/><path d="M6 4h7a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1z" stroke="currentColor" stroke-width="1.2" fill="none"/></svg>';
    button.innerHTML = defaultHTML;

    button.addEventListener('click', function () {
      var text = codeEl.textContent;

      function showCopied() {
        button.classList.add('copied');
        button.textContent = 'Copied';
        setTimeout(function () {
          button.classList.remove('copied');
          button.innerHTML = defaultHTML;
        }, 1500);
      }

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(showCopied);
      } else {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (e) {}
        document.body.removeChild(ta);
        showCopied();
      }
    });

    block.appendChild(button);
  });
});
