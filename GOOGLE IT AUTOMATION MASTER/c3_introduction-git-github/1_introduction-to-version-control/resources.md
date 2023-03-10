## Resources

diff
diff is used to find differences between two files. On its own, it’s a bit hard to use; instead, use it with diff -u to find lines which differ in two files:

diff -u
diff -u is used to compare two files, line by line, and have the differing lines compared side-by-side in the same output.

Patch
Patch is useful for applying file differences. See the below example, which compares two files. The comparison is saved as a .diff file, which is then patched to the original file!

There are some other interesting patch and diff commands such as patch -p1, diff -r !

<div ><p><span><span>Check out the following links for more information:</span></span></p><p><span><span>The </span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/Documentation/process/submitting-patches.rst?id=HEAD"><span><u><span>Linux kernel documentation</span></u></span></a></span><span><span> itself, as well as </span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="http://stopwritingramblingcommitmessages.com/"><span><u><span>impassioned</span></u></span></a></span><span><span> opinions from other </span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="https://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message"><span><u><span>developers</span></u></span></a></span><span><span>.&nbsp;</span></span></p><p><span><span>You can check out "</span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="https://help.github.com/articles/setting-your-email-in-git/"><span><u><span>Setting your email in Git</span></u></span></a></span><span><span>" and "</span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="https://help.github.com/articles/keeping-your-email-address-private/"><span><u><span>Keeping your email address private</span></u></span></a></span><span><span>" on the GitHub help site for how to do this.  </span></span></p></div>

<br>

- http://man7.org/linux/man-pages/man1/diff.1.html
- http://man7.org/linux/man-pages/man1/patch.1.html
- [Git download page](https://git-scm.com/downloads)
- [Git installation instructions for each platform](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- https://git-scm.com/doc
- https://www.mercurial-scm.org/
- https://subversion.apache.org/
- https://en.wikipedia.org/wiki/Version_control
- [Linux kernel documentation](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/Documentation/process/submitting-patches.rst?id=HEAD)
- [impassioned opinions from other developers](http://stopwritingramblingcommitmessages.com/)
- https://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message).
- [Setting your email in Git](https://help.github.com/articles/setting-your-email-in-git/)
- [Keeping your email address private](https://help.github.com/articles/keeping-your-email-address-private/)
