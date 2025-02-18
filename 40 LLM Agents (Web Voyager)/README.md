LLM Agents (WebVoyager)

The WebVoyager concept described in this paper: https://arxiv.org/abs/2401.13919

Goals:
- Implement am LLM agent that can perform web-browsing tasks requested by user
- Use Set-of-Marks image annotations to facilitate navigation through a screenshot of a webpage
- Implement 6 ways (called tools) for the agent to navigate the browser:
  - Click
  - Type
  - Scroll
  - Wait
  - Go back
  - Go to search engine
- Use `langgraph` to create a graph defining all possible steps of the workflow and their relationship.
