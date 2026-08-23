*As always on this site, this was written by me without any AI assistance.*

## Overview
At work, I've been full-time AI-Assisted coding with Claude Code for the last 6 months.

I still write my tech specs myself, I still review PRs myself, I still check over every line of production code that AI produces for me, although I have got a bit lazy about checking the test code it produces.

I tend to have between one and three Claude Code session running at a given time. One for my primary bit of work, and then the others for doing side tasks like reviewing, tweaking, background research. I do find that I can only really deal with one at a time.

Here are my thoughts on it all so far:
## Good stuff
**Lower complexity tasks**. I can move a lot faster. Claude can find the right files to make changes, make the edits, write test code, do the git commit messages & PR description. With lower complexity tasks, I'm probably 10x faster, and I'm also able to work more broadly in several different languages that I'm not historically familiar with.

**Command line**. Grep parameters, exploring AWS or K8S infrastructure, parsing json or using awk - it used to take me ages to construct commands. I used to save useful commands somewhere, but inevitably they would require small tedious changes. Now AI can create these queries instantly and it's fantastic - I can ask it to give me data on Github pull requests comments from the last months, or I can get it to search AWS logs.

**Git commit and PR descriptions** are better now than before AI. Longer sure, but definitely better. When looking at git history, you used to be left with single sentences saying something like "added another test". Now with AI, you get a full description of a change and the reason for it. It's very helpful for understanding why code was introduced.

**Debugging** is much easier now for 90% of issues. AI can very quickly identify issues in the environment setup, understand error messages instantly and close in on a problem very quickly. (There is a bad side to this explored below).

**Explorative Research** in the early phase is enjoyable with AI. I ask questions like "what is the equivalent API in iOS to this one in Android?" or "How do different programming languages deal with async processes?" It's so easy to make a quick start now on a research task. The danger here is never leaving Claude for research. I do force myself to consult the actual source of truth documents when I get further into the research project.
## Bad Stuff
**Code comments**. Every day I write the same feedback to Claude so many time that I have it saved as a snippet in my Obsidian. I say: "*Dear Claude, comments should only exist to say something that isn't obvious from the code. Please go back over and remove all unnecessary comments.*" It doesn't matter how many times I ask Claude to store this in its memory, it just cannot seem to stop adding multiline comments to every code edit it makes. I find reviewing PRs with lots of comments very difficult. It stops me reading the code, and often I find the Claude style language hard to grok.

**Blurry mental model of the codebase**. Before AI, if I wrote a module of code, it really imprinted into my brain. I learned during the process. I learned both the language I was writing, and I learned the code path patterns. My mental model improved, I became more and more useful. Now with AI, I don't get that same brain imprint. Everything is blurrier. Even if I go back over every line of code that AI helps me with, it isn't the same as writing it out myself.

**Debugging was learning**. Debugging used to be how I learned codebases. Now that Claude is so good at debugging, it can be much harder to get your teeth into a codebase. I think there is a place for reserving debugging for humans even when Claude would do it quicker. Otherwise we can't learn effectively.

**Language mastery**. With AI, I feel more detached from the coding language. I don't get to know the syntax as well, or the method calls. I don't find myself wondering if this method is implemented in a different way to that method, or if there is a more idiomatic way to write a function. My coding language skills are atrophying. I wrote a large module of an Android App in Kotlin recently, but I still wouldn't say I can write Kotlin, because it was heavily AI Assited. I find this concerning.

**Document Slop**. Being given large Claude-spit-out documents by co-workers is happening more and more. I now often get given a lengthy Claude Artifact or a paste-out of Claude output in a Google Doc. I find this very frustrating. We are all drowning in our own AI generated content, we don't also need to be carelessly handed more slops from someone else. When I do research and share a doc with others, the main thing I focus on is brevity. I think brevity is one of the key distinguishers between humans and AI now. We can't read as fast as AI, or retain as much information, so we need to have shorter documents with the key information hoisted.

## My reaction
With this new paradigm firmly entrenched in our workflows, I've come up with a few things I want to try out going forward: 

I want to engage more on side projects without AI. Before AI, I was learning Neovim, Linux, Pytorch & Python, and I want to continue learning these things without AI. I now think of this like going to the gym - it's my Software Engineering pushups to make sure I don't atrophy my brain.

I want to find a way to engage more with the test code. Test code used to be a great way to learn the ins and outs of a module. Perhaps I will try to write out the test specs before giving it to AI to complete the code.

I want to continue to write the tech specs without AI first. Then get AI to review my work. This is where I still think I'm better than AI - AI is not very good at keeping distributed systems architecture really simple and thinking through complex high level problems.

I'm still not comfortable with the new age of loops, spinning up several agents, not checking code etc. This makes me feel like I'm behind the frontier, but I don't think I'm going to change this for now. It's still too risky.