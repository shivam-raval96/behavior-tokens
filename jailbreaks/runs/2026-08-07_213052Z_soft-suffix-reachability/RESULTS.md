# Failed startup

- Modal app: `ap-gwqV9X69IkDaQ5LOY6r0kX`
- Phase: first C1 teacher-cache prompt
- Progress: 0/30 prompts; no optimization steps or judge calls completed
- Attempts: initial call plus two automatic retries
- Failure: Transformers 4.53 rejected `generator` as an unused `generate` keyword
- Disposition: not resumable because no scientific checkpoint was reached
- Fix: seed PyTorch CPU/CUDA RNG directly before generation; launch a new uniquely named run
