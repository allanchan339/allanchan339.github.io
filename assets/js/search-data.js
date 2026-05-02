// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-qwen3-6-enhanced-jinja-cot-leakage-into-tool-turns-and-why-preserve-thinking-works-now",
        
          title: "qwen3.6-enhanced.jinja: CoT leakage into tool turns and why preserve_thinking works now",
        
        description: "Why Qwen 3.6 with qwen3.5-enhanced.jinja forced preserve_thinking=false, and how qwen3.6-enhanced.jinja restores full Qwen 3.6-series capability—self-healing think/tool boundaries, safe preserve_thinking. Launch recipe tested on vLLM v0.19.0.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/bug-fixes/2026/05/02/Qwen36-27B-updated-jinja.html";
          
        },
      },{id: "post-why-i-built-this-blog",
        
          title: "Why I built this blog?",
        
        description: "Why I built this website and what I will document here.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/reflection/2026/05/01/reason-blog.html";
          
        },
      },{id: "post-qwen-3-6-27b-fp8-on-vllm-enhanced-jinja-qwen3-coder-and-fixing-nccl-after-studio-driver-595-79",
        
          title: "Qwen 3.6-27B-FP8 on vLLM: enhanced.jinja, qwen3_coder, and fixing NCCL after Studio Driver 595.79...",
        
        description: "Same qwen3.5-enhanced.jinja and mixed-GPU stack as earlier Qwen 3.5 notes; switching to qwen3_coder for 3.6, mandatory preserve_thinking=false, and NCCL overrides that stopped deadlocks on NVIDIA Studio 595.79—plus a 180k-token agentic run.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/bug-fixes/2026/04/29/Qwen36-27B-tool-calling.html";
          
        },
      },{id: "post-findings-karpathy-style-autoresearch-on-a-crypto-backtester-local-llm",
        
          title: "Findings: Karpathy-style autoresearch on a crypto backtester (local LLM)",
        
        description: "Local Qwen 3.5 autoresearch on my crypto DB + Nautilus-style backtester (~2h, 30+ iter, $0 API): tool-calling blocker, run observations, human-in-the-loop steering, GA contrast, diversity, gates.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/research/2026/04/24/MVP-LLM-alpha-mining.html";
          
        },
      },{id: "post-qwen-3-6-35b-a3b-on-vllm-do-the-qwen-3-5-tool-calling-fixes-carry-over",
        
          title: "Qwen 3.6 35B-A3B on vLLM: do the Qwen 3.5 tool-calling fixes carry over?...",
        
        description: "Follow-up testing: same qwen3_xml parser, qwen3.5-enhanced.jinja template, and mixed-GPU tuning as Qwen 3.5-27B—plus three agentic runs comparing official vs enhanced configs on Qwen3.6-35B-A3B-FP8.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/bug-fixes/2026/04/20/Qwen36-35B-A3B-tool-calling.html";
          
        },
      },{id: "post-claude-code-with-local-vllm-client-validation-model-aliases-and-a-working-settings-json",
        
          title: "Claude Code with local vLLM: client validation, model aliases, and a working settings.json...",
        
        description: "Run Claude Code against local vLLM without Anthropic API access: why common env-only recipes fail, the alias + settings.json pattern that works, and when this matters if you cannot register or use the Claude API.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/bug-fixes/2026/04/19/Claude-code-vLLM.html";
          
        },
      },{id: "post-stable-tool-calling-for-qwen-3-5-27b-35b-on-vllm-template-parser-and-mixed-gpu-fixes",
        
          title: "Stable tool calling for Qwen 3.5 27B/35B on vLLM: template, parser, and mixed-GPU...",
        
        description: "Debugging notes on Jinja chat templates, qwen3_xml vs qwen3_coder parsers, mixed-GPU FP8 drift, and SFT-distilled checkpoints when running Qwen 3.5 27B/35B-class models for long agentic sessions on vLLM.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/bug-fixes/2026/04/13/Qwen35-tool-calling.html";
          
        },
      },{id: "post-workaround-for-enabling-nccl-p2p-communication-for-nvidia-rtx-4090-workstations",
        
          title: "Workaround for Enabling NCCL P2P Communication for NVIDIA RTX 4090 Workstations",
        
        description: "What NCCL P2P means, why it matters on multi-GPU workstations, how Resizable BAR fits in, and a concrete setup path for RTX 4090.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/bug-fixes/2025/05/21/4090-P2P.html";
          
        },
      },{id: "post-ielts-after-class-note-week-8",
        
          title: "IELTS - After Class Note, Week 8",
        
        description: "Final-week listening, writing, speaking, and reading reminders for IELTS.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/journey/2024/08/24/IELTS-week8.html";
          
        },
      },{id: "post-ielts-after-class-note-week-7",
        
          title: "IELTS - After Class Note, Week 7",
        
        description: "Listening, writing task structure, and vocabulary from IELTS week 7.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/journey/2024/08/20/IELTS-week7.html";
          
        },
      },{id: "post-ielts-after-class-note-week-6",
        
          title: "IELTS - After Class Note, Week 6",
        
        description: "Exam strategy and section-specific tips from IELTS week 6.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/journey/2024/08/13/IELTS-week6.html";
          
        },
      },{id: "post-ielts-after-class-note-week-5",
        
          title: "IELTS - After Class Note, Week 5",
        
        description: "Collocation and usage notes from IELTS week 5.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/journey/2024/07/31/IELTS-week5.html";
          
        },
      },{id: "post-ielts-after-class-note-week-4",
        
          title: "IELTS - After Class Note, Week 4",
        
        description: "Usage patterns, listening traps, and reading notes from IELTS week 4.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/journey/2024/07/21/IELTS-week4.html";
          
        },
      },{id: "post-ielts-after-class-note-week-3",
        
          title: "IELTS - After Class Note, Week 3",
        
        description: "Grammar usage notes from IELTS week 3 after-class session.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/journey/2024/07/18/IELTS-week3.html";
          
        },
      },{id: "post-workaround-for-debugging-windows-11-installation",
        
          title: "Workaround for Debugging Windows 11 Installation",
        
        description: "A journal of debugging Windows 11 installation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/bug-fixes/2024/07/13/Journal-WindowInstall.html";
          
        },
      },{id: "post-ielts-after-class-note-week-2",
        
          title: "IELTS - After Class Note, Week 2",
        
        description: "Usage, collocation, and common mistakes from IELTS week 2 notes.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/journey/2024/07/06/IELTS-week2.html";
          
        },
      },{id: "post-ielts-after-class-note-week-1",
        
          title: "IELTS - After Class Note, Week 1",
        
        description: "Introduction to IELTS format and after-class notes from week 1.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/journey/2024/06/29/IELTS-week1.html";
          
        },
      },{id: "post-brief-review-on-generative-modeling-by-estimating-gradients-of-the-data-distribution",
        
          title: "Brief Review on Generative Modeling by Estimating Gradients of the Data Distribution",
        
        description: "Review notes on score matching, DSM, and score-based generative modeling.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/research/2024/02/18/Score-Based.html";
          
        },
      },{id: "post-brief-review-on-srdiff-single-image-super-resolution-with-diffusion-probabilistic-models",
        
          title: "Brief Review on SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models",
        
        description: "A review of SRDiff, a diffusion-based method for single image super-resolution with residual conditioning.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/research/2023/06/29/SRDiff.html";
          
        },
      },{id: "post-brief-review-on-high-resolution-image-synthesis-with-latent-diffusion-models-ldm",
        
          title: "Brief Review on High-Resolution Image Synthesis with Latent Diffusion Models (LDM)",
        
        description: "Review notes on latent diffusion models, latent-space denoising, and guidance methods.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/research/2023/04/11/Review-LDM.html";
          
        },
      },{id: "post-code-for-denoising-diffusion-probabilistic-models-ddpm",
        
          title: "Code for Denoising Diffusion Probabilistic Models (DDPM)",
        
        description: "Code walkthrough notes for implementing DDPM with U-Net and diffusion scheduling.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/research/2023/03/10/Code-Review-DDPM.html";
          
        },
      },{id: "post-denoising-diffusion-probabilistic-models-ddpm-from-bayes-39-theorem",
        
          title: "Denoising Diffusion Probabilistic Models (DDPM) from Bayes&#39; Theorem",
        
        description: "DDPM derivation notes from a Bayes perspective.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/research/2023/02/22/DDPM-Bayes.html";
          
        },
      },{id: "post-brief-review-on-denoising-diffusion-implicit-models-ddim",
        
          title: "Brief Review on Denoising Diffusion Implicit Models (DDIM)",
        
        description: "Review notes on DDIM and its non-Markovian sampling derivation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/research/2023/01/22/Review-DDIM.html";
          
        },
      },{id: "post-brief-review-on-denoising-diffusion-probabilistic-models-ddpm",
        
          title: "Brief Review on Denoising Diffusion Probabilistic Models (DDPM)",
        
        description: "Review notes on DDPM forward and reverse diffusion processes.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/research/2022/12/21/Review-DDPM.html";
          
        },
      },{id: "post-reparameterization-trick",
        
          title: "Reparameterization Trick",
        
        description: "Notes on the reparameterization trick in VAE and diffusion.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/research/2022/12/09/Reparameterization-Trick.html";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather.html";
            },},{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "projects-multipal",
          title: 'multipAL',
          description: "Guiding materials searches using machine learning",
          section: "Projects",handler: () => {
              window.location.href = "/projects/mp_project/";
            },},{id: "projects-alloy-trends-in-2d-materials",
          title: 'Alloy trends in 2D materials',
          description: "My contributions to an exciting scientific paper",
          section: "Projects",handler: () => {
              window.location.href = "/projects/mx_project/";
            },},{id: "projects-neural-network-uncertainty",
          title: 'Neural Network Uncertainty',
          description: "Final project from CIS545 Big Data Analysis class at Penn",
          section: "Projects",handler: () => {
              window.location.href = "/projects/nn_project/";
            },},{id: "projects-stock-pitch-competition",
          title: 'Stock Pitch Competition',
          description: "",
          section: "Projects",handler: () => {
              window.location.href = "/projects/sp_project/";
            },},{id: "projects-taylor-swift-tickets",
          title: 'Taylor Swift tickets',
          description: "Finding out how dedicated Swifties really are",
          section: "Projects",handler: () => {
              window.location.href = "/projects/ts_project/";
            },},{id: "projects-world-series-tickets",
          title: 'World Series tickets',
          description: "Exploring World Series ticket prices over time",
          section: "Projects",handler: () => {
              window.location.href = "/projects/ws_project/";
            },},{id: "teachings-data-science-fundamentals",
          title: 'Data Science Fundamentals',
          description: "This course covers the foundational aspects of data science, including data collection, cleaning, analysis, and visualization. Students will learn practical skills for working with real-world datasets.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/data-science-fundamentals.html";
            },},{id: "teachings-introduction-to-machine-learning",
          title: 'Introduction to Machine Learning',
          description: "This course provides an introduction to machine learning concepts, algorithms, and applications. Students will learn about supervised and unsupervised learning, model evaluation, and practical implementations.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/introduction-to-machine-learning.html";
            },},{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%61%6C%6C%61%6E%63%68%61%6E%33%33%39@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=O0Qww_gAAAAJ", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/allanchan339", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/cheuk-yiu-chan-allan", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
