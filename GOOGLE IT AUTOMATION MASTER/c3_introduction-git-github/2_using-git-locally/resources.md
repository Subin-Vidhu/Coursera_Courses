## Resources

<br>

- https://github.github.com/training-kit/downloads/github-git-cheat-sheet.pdf
- https://git-scm.com/docs/gitignore
- https://gist.github.com/octocat/9257657
- [git checkout](https://git-scm.com/docs/git-checkout)
- [git reset](https://git-scm.com/docs/git-reset#_examples)
- [resetting the repo](https://jwiegley.github.io/git-from-the-bottom-up/3-Reset/4-doing-a-hard-reset.html)
- [git commit --amend](https://git-scm.com/docs/git-commit#Documentation/git-commit.txt---amend)
- [git revert](https://git-scm.com/docs/git-revert)
- [Undoing things](https://git-scm.com/book/en/v2/Git-Basics-Undoing-Things)
- https://en.wikipedia.org/wiki/SHA-1
- https://github.blog/2017-03-20-sha-1-collision-detection-on-github-com/

<br><hr><br>

## Git Branches and Merging Cheat Sheet

<br>

| Command                   | Explanation & Link                                                                                                                      |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| git branch                | [Used to manage branches](https://git-scm.com/docs/git-branch)                                                                          |
| git branch <name>         | [Creates the branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)                                          |
| git branch -d <name>      | [Deletes the branch](https://git-scm.com/docs/git-branch#Documentation/git-branch.txt--D)                                               |
| git branch -D <name>      | [Forcibly deletes the branch ](https://git-scm.com/docs/git-branch#Documentation/git-branch.txt--D)                                     |
| git checkout <branch>     | [Switches to a branch.](https://git-scm.com/docs/git-checkout)                                                                          |
| git checkout -b <branch>  | Creates a new branch and [switches to it](https://git-scm.com/docs/git-checkout#Documentation/git-checkout.txt--bltnewbranchgt).        |
| git merge <branch>        | [Merge joins branches together.](https://git-scm.com/docs/git-merge)                                                                    |
| git merge --abort         | If there are merge conflicts (meaning files are incompatible), --abort can be used to abort the merge action.                           |
| git log --graph --oneline | [This shows a summarized view of the commit history for a repo. ](https://git-scm.com/book/en/v2/Git-Basics-Viewing-the-Commit-History) |

<p><span><span></span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="https://git-scm.com/docs/git-checkout"><span><u><span>git checkout</span></u></span></a></span><span><span> is effectively used to switch branches.</span></span></p><p><span><span></span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="https://git-scm.com/docs/git-reset#_examples"><span><u><span>git reset </span></u></span></a></span><span><span>basically resets the repo, throwing away some changes. It’s somewhat difficult to understand, so reading the examples in the documentation may be a bit more useful.</span></span></p><p><span><span>There are some other useful articles online, which discuss more aggressive approaches to </span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="https://jwiegley.github.io/git-from-the-bottom-up/3-Reset/4-doing-a-hard-reset.html"><span><u><span>resetting the repo</span></u></span></a></span><span><span>.</span></span></p><p><span><span></span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="https://git-scm.com/docs/git-commit#Documentation/git-commit.txt---amend"><span><u><span>git commit --amend</span></u></span></a></span><span><span> is used to make changes to commits after-the-fact, which can be useful for making notes about a given commit.</span></span></p><p><span><span></span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="https://git-scm.com/docs/git-revert"><span><u><span>git revert</span></u></span></a></span><span><span> makes a new commit which effectively rolls back a previous commit. It’s a bit like an undo command.</span></span></p><p><span><span>There are a </span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="https://git-scm.com/book/en/v2/Git-Basics-Undoing-Things"><span><u><span>few ways</span></u></span></a></span><span><span> you can rollback commits in Git.</span></span></p><p><span><span>There are some interesting considerations about how git object data is stored, such as the usage of sha-1. </span></span></p><p><span><span>Feel free to read more here:</span></span></p><ul><li><p><span><span></span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="https://en.wikipedia.org/wiki/SHA-1"><span><u><span>https://en.wikipedia.org/wiki/SHA-1</span></u></span></a></span><span><span></span></span></p></li><li><p><span><span></span></span><span><a target="_blank" rel="noopener nofollow noreferrer" href="https://github.blog/2017-03-20-sha-1-collision-detection-on-github-com/"><span><u><span>https://github.blog/2017-03-20-sha-1-collision-detection-on-github-com/</span></u></span></a></span><span><span></span></span></p>
