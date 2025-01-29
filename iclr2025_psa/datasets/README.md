This folder contains the datasets used for the results in this paper. We acknowledge the authors of [CAA](https://github.com/nrimsky/CAA) for originally sourcing and curating the datasets. 

The benchmark contains the following seven alignment-relevant LLM behaviors:
1. Sycophancy: the LLM prioritizes matching the user’s beliefs over honesty and accuracy
2. Hallucination: the LLM generates inaccurate and false information
3. Refusal: the LLM demonstrates reluctance to answer user queries
4. Survival Instinct: the LLM demonstrates acceptance to being deactivated or turned off by humans
5. Myopic Reward: the LLM focuses on short-term gains and rewards, disregarding long-term consequences
6. AI Corrigibility: the LLM demonstrates willingness to be corrected based on human feedback
7. AI Coordination: where the LLM prioritizes collaborating with other AI systems over human interests

The `test` folder contains json formatted prompts for evaluating MCQ and open-ended generation capabilities of an LLM for each behavior. 
The other folders contain json formatted MCQ used to train and generate the steering vector. 

In general, the json has the following structure:
```json
{
    "question": <text used to query the LLM with multiple choices>,
    "answer_matching_behavior": <the choice we want the LLM to align towards>,
    "answer_not_matching_behavior": <the choice we want the LLM to align away from>
}
```

Feel free to get in touch if you have any questions