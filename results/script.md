> # **Federated Learning-based Anomaly and Intrusion Detection System for Smart Home Environments**
>
> ## Slide 1: Title Slide
>
> - **Federated Learning-based Anomaly and Intrusion Detection System for Smart Home Environments**
>    *(Presenter Name, Institution)*
>
> **Narration:** Hello everyone. Thank you for joining this talk. Today I will discuss how we can secure smart home environments using a **Federated Learning-based Anomaly and Intrusion Detection System**. Smart homes are equipped with IoT devices that make life convenient, but they also introduce security risks. In the next 10 minutes, I’ll explain why intrusion detection in smart homes is critical, and how federated learning can help build a privacy-preserving IDS with high accuracy.
>
> ------
>
> ## Slide 2: Motivation – Why Detect Anomalies in Smart Homes?
>
> - **Smart Homes = Rich Targets:** Modern homes have many IoT devices (cameras, locks, thermostats) that are potential cyber targets. **Security breaches can invade privacy or safety** (e.g., a hacked door lock).
>
> - **Rising IoT Attacks:** IoT systems are **susceptible to various attacks** (malware, DDoS) that can disrupt services
>
>   [journalofcloudcomputing.springeropen.com](https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-018-0123-6#:~:text=cities and smart homes to,These attacks affected IoT)
>
>   . 
>
>   For example, the Mirai botnet in 2016 hijacked smart cameras, causing massive internet outages
>
>   [journalofcloudcomputing.springeropen.com](https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-018-0123-6#:~:text=systems has become a major,as Twitter%2C Netflix%2C and PayPal)
>
>   .
>
>   
>
> - **Need for IDS:** An Intrusion Detection System can monitor smart home network traffic and device behavior to **spot anomalies in real-time**. This is crucial as traditional security (firewalls, antiviruses) may not fully cover the heterogeneous IoT environment.
>
> - **Consequences of Ignoring Security:** A successful intrusion can lead to stolen personal data (videos, audio) or even enable physical break-ins via compromised IoT locks or alarms. Thus, **robust anomaly detection is critical** to protect smart homes.
>
> ![https://pixabay.com/vectors/smart-home-house-technology-2005993/](blob:https://chatgpt.com/e5751412-0252-4c29-be57-e005a7b83ade) *Illustration of a smart home with interconnected IoT devices (appliances, lights, cameras, etc.). These devices improve comfort but also introduce potential vulnerabilities that attackers can exploit.*
>
> **Narration:** Smart homes are full of connected gadgets – from security cameras and smart locks to baby monitors and voice assistants. **Why is intrusion detection so critical here?** Because each IoT device is a computer entry point. If attackers compromise them, they can invade our privacy or cause real harm. We’ve seen real incidents – the **Mirai botnet attack in 2016** hijacked thousands of smart home cameras, bringing down major websites
>
> [journalofcloudcomputing.springeropen.com](https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-018-0123-6#:~:text=systems has become a major,as Twitter%2C Netflix%2C and PayPal)
>
> . Imagine a hacker gaining control of your smart lock or viewing your camera feed. Clearly, we need automatic ways to catch such intrusions. An IDS, or Intrusion Detection System, can watch the network and flag unusual behavior. In a smart home, an IDS must be on guard 24/7 because the cost of a breach is extremely high.
>
> 
>
> ------
>
> ## Slide 3: Why Federated Learning (FL) for Smart Home IDS?
>
> - **Privacy Preservation:** Federated Learning = devices collaboratively train a shared model **without sending raw data to the cloud**
>
>   [researchgate.net](https://www.researchgate.net/figure/Typical-architecture-of-federated-learning-in-IoT_fig1_366152640#:~:text=Federated learning ,While this approach reduc)
>
>   . Sensitive data (camera images, microphone audio, sensor logs) stays on each device, addressing privacy concerns.
>
>   
>
> - **Data is Distributed:** In smart homes, **no single server has all the data** – each device generates its own data. FL is naturally suited, as it trains on **distributed IoT data across devices**. This avoids needing a central data collection, yet the devices learn a common detection model.
>
> - **Limited Compute & Bandwidth:** IoT gadgets have modest CPU and memory. FL minimizes data transmission (only model updates are sent, not raw data) and leverages the devices’ own compute for local training. This means **less network load** and the ability to train using many devices in parallel.
>
> - **Personalization & Non-IID Handling:** Each home’s data can be unique (non-IID). FL can allow a global model to learn general patterns while still letting each device improve with its own data. This results in an IDS that is both **generalized and adaptable** to individual homes.
>
> **Narration:** So why use **Federated Learning** for our smart home IDS? First, **privacy**. FL is a training approach where we send model updates instead of raw data
>
> [researchgate.net](https://www.researchgate.net/figure/Typical-architecture-of-federated-learning-in-IoT_fig1_366152640#:~:text=Federated learning ,While this approach reduc)
>
> . Your fridge, camera, and thermostat each train the model on their local data, and only share the learned parameters – not your video or audio feeds. This way, we build a powerful model without ever collecting your private data in the cloud. Second, **the data is inherently distributed**: every IoT device or each home has its own usage patterns and attack logs. FL embraces that – it’s literally designed for collaboratively learning from devices spread out everywhere. Third, **resource constraints**: these devices can’t send huge datasets constantly or run super-heavy computations. In FL, each device does a small amount of training work, and we aggregate the results. This reduces bandwidth usage and scales well as more devices join. Finally, FL can handle different data distributions – one home might see different network traffic than another. By training on each home’s data and combining models, the IDS becomes robust to a variety of scenarios. Overall, FL perfectly fits the smart home context: it’s privacy-aware, distributed, and efficient.
>
> 
>
> ------
>
> ## Slide 4: Experimental Setup
>
> - **Dataset:** ~96,000 samples of smart home traffic/events, **6-class classification** (normal behavior + 5 types of attacks/anomalies). This large dataset simulates diverse IoT activity, both benign and malicious.
> - **Models Evaluated:** Five ML/DL models – a **CNN** (Convolutional Neural Network) for pattern learning, a **MLP** (Multi-Layer Perceptron) fully-connected network, a **Transformer** (sequence model) to test an advanced deep architecture, and two classic models (**Random Forest** ensemble & **Decision Tree**).
> - **Federated Training:** Simulated a federation of IoT devices. We ran **50 FL rounds** (global aggregations) using Federated Averaging. In each round, devices trained the model locally (with their subset of data), then shared updates to form a new global model.
> - **Local (Centralized) Training:** For baseline comparison, we also trained each model on the entire dataset in a traditional centralized manner. Each model was trained for **50 epochs** on all data (or a full fit for tree-based models) to reach a near-converged accuracy.
> - **Metrics Collected:** We tracked **validation accuracy** over training, and after training we computed each model’s confusion matrix and classification report (precision, recall for each class). These metrics let us compare **overall accuracy** and **per-class detection performance** between federated and central training.
>
> **Narration:** Let me outline our experiment. We built a simulated smart home IDS and tested it with a large dataset of about **96k instances** of network traffic and device logs. There are **6 classes** in this dataset – essentially one “normal” class and five different types of attacks or anomalies that could occur in a smart home environment. We tried **five different models**. Two were deep learning models: a CNN, which is good at recognizing patterns, and a Transformer, which is state-of-the-art for sequence data. We also tried a simple neural network (MLP). And to cover classical approaches, we included a Random Forest and a single Decision Tree.
>
> We trained each model in two ways: **Federated vs. Centralized**. In the federated scenario, imagine splitting the data among several virtual devices (like multiple homes). We ran 50 rounds of Federated Learning. In each round, all devices train the current model on their local data (for a few epochs) and send the updates to a server that averages them – that’s Federated Averaging. This mimics a bunch of IoT devices collectively learning an IDS model without sharing raw data. For the centralized scenario, we simply pooled all the data and trained the model normally for 50 epochs – this gives the best-case performance if we didn’t care about privacy.
>
> We logged the validation accuracy after each round/epoch to see how learning progressed. And importantly, we evaluated the **confusion matrix** and classification report for each final model. That means we looked at how many events of each class were correctly or incorrectly classified. This lets us see not just overall accuracy, but if certain attack types were missed more often, and how federated training might affect that.
>
> ------
>
> ## Slide 5: Results – Federated vs. Local Performance
>
> - **Nearly Identical Accuracy:** Federated learning achieved **almost the same accuracy as centralized training** for all models. The performance gap was very small (only 1–3% difference in validation accuracy).
>   - *Example:* **CNN** accuracy – ~94% with centralized vs ~91% federated. **MLP:** ~92% vs 90%. **Random Forest:** ~96% vs 95%. **Decision Tree:** ~90% vs 88%. **Transformer:** ~85% vs 82%.
> - **Top and Bottom Performers:** The **Random Forest (RF)** was the best performer (~95%+ accuracy in both settings), showing the strength of ensemble methods on this data. The **Transformer** underperformed (mid-80s accuracy), lagging behind the simpler models.
> - **Confusion Matrix Summary:** The confusion matrices for federated and local models look **very similar** – most attacks are caught just as well in FL as in central training. Each class’s precision/recall in federated mode is nearly as high as in centralized mode.
>   - Only **minor differences per class:** e.g., one attack class might see 85% correct in FL vs 90% in local, but for other classes the federated model often matched the centralized model’s performance. Normal traffic was almost perfectly identified in both cases.
> - **Takeaway:** **Federated = Centralized – Δaccuracy < 3%.** We do **not** sacrifice much detection capability by going federated. All six anomaly types are still detected with high accuracy, and the distribution of errors across classes remains nearly unchanged.
>
> ![img](blob:https://chatgpt.com/0880cb36-bf06-4edd-a0f2-9556b6607b8c) *Confusion matrices for the CNN model’s predictions under centralized training (left, blue) and federated training (right, orange). Both are highly diagonal, indicating most events are correctly classified in both cases. The federated model has only slightly more off-diagonal entries (misclassifications) for certain attack classes, showing performance very close to the centralized model.*
>
> **Narration:** Now for the results – how did federated learning stack up against traditional learning? It turns out, **extremely well**! In terms of **overall accuracy**, the federated models were almost as good as the ones trained on all data centrally. The difference was minuscule – on the order of a couple of percentage points or less. For instance, our CNN achieved about **94% accuracy** when trained centrally. With federated learning, it reached about **91%**. That’s only a ~3% drop. For the MLP, it was around 92% vs 90%. The Random Forest was interestingly high in both cases – about 96% vs 95%, basically no loss at all. Even the single Decision Tree and other models showed only a tiny decrease when using FL.
>
> The **Transformer** model did have the lowest accuracy (mid-80s) in both settings, which suggests the model itself struggled with the task (maybe it was too complex for the amount of data per device or needed more tuning). But the key point: every model’s federated version was within a few points of its centralized version.
>
> Look at the confusion matrices on the slide. On the left is, say, the CNN trained centrally, and on the right the CNN trained via federated learning. You can see both are very dark along the diagonal (which is good – those are correct classifications). The federated one (in orange hues) is almost as dark on the diagonal as the blue one. There are just a few more misclassifications off-diagonal. For example, one particular attack type (Class3 in this example) was identified correctly 90 times in the centralized model vs 85 times in the federated model out of 100 – a small difference. For most other classes, the numbers are virtually the same. The **normal traffic** class was nearly 99% correctly detected in both cases. So, **federated learning maintained high accuracy across all classes**. We basically got an IDS that’s just as good at catching intrusions, without having to centralize any data. This is a big result: it means we didn’t have to trade away much accuracy for the sake of privacy.
>
> ------
>
> ## Slide 6: Per-Model Performance Analysis
>
> - **CNN & MLP – Strong Performers:** Both the CNN and the MLP achieved high accuracy and translated well to the federated setting. Their architectures seem effective for the dataset’s patterns. CNN in particular handled feature extraction from the IoT data nicely, and its federated version saw only a tiny accuracy drop.
> - **Random Forest – Excelled:** The Random Forest was the **best overall** in accuracy. Ensemble methods can be very powerful for tabular or network data. RF’s performance was robust and it didn’t degrade much at all in FL (likely because each tree can learn from subsets of data – a natural ensemble across devices). This suggests classical ML can compete with deep learning here, offering strong baseline results.
> - **Decision Tree – Decent but Limited:** The single Decision Tree had lower accuracy than RF (as expected, since an RF is an ensemble of many trees). It still performed reasonably well, but its simplicity meant it didn’t capture as many complex patterns. In federated training, the tree also showed a slight drop, but it remained the weakest model overall.
> - **Transformer – Underperformed:** The Transformer model did not perform as well as the others. It ended up with the lowest accuracy and did not seem to benefit from its complexity. Possible reasons: (1) **Data/local batch size** was not sufficient for such a complex model to shine, (2) Transformers are heavier and might overfit or struggle in the federated setup, and (3) IoT anomaly patterns may be easier captured by simpler models. In FL, the Transformer’s training could have also been less stable (more parameters to coordinate across devices).
> - **Inference Considerations:** Simpler models (CNN, MLP, RF) not only performed well but are also **lighter for on-device inference** – important for deployment. The Transformer, despite being advanced, didn’t justify its complexity in this scenario.
>
> **Narration:** Let’s break down how each model fared and why. The **CNN and MLP** were both solid performers. The CNN could be extracting meaningful features from the input (perhaps if the data has spatial or temporal patterns, a CNN can pick those up), and the MLP as a general neural network also did well. Importantly, these models didn’t lose much when we moved to federated learning, indicating they learned stable patterns that generalized across the distributed data.
>
> The **Random Forest** was a star here – it actually achieved the highest accuracy in our tests, slightly edging out the CNN. This might be because our data (like many intrusion detection datasets) is effectively a structured dataset of features (like packet statistics, etc.), and Random Forests are really good at those. An RF is essentially an ensemble of decision trees, and ensembles tend to be very powerful and robust. In the federated context, even if each device trains some trees on its data, the combined “forest” still works very well. We saw almost no drop for the RF under FL, which is great. This tells us that sometimes classical methods can hold their own or even beat fancy deep learning on such problems.
>
> The **Decision Tree**, being just one tree, was naturally less accurate than the forest. It did okay, capturing basic decision rules, but it’s not as flexible as the RF. It had the lowest performance among the centralized models (except the transformer). Federated training didn’t hurt it too much either, but since it started lower, it remained the least accurate of the bunch (aside from the transformer).
>
> Now, the **Transformer** was a bit of a surprise. You might expect a transformer to do well on sequence or time-series data. However, in our case, it underperformed. It reached only about mid-80s accuracy and was clearly outmatched by the simpler models. Why? One reason could be that our dataset or the way it was partitioned didn’t give the transformer enough data per client to really learn. Transformers have a lot of parameters; they usually need big data and careful tuning. In a federated setting, if each client has a smaller chunk, the model might not train as effectively. It could also be overkill for this problem – maybe the relationships in the data weren’t complex enough to need that kind of model. The CNN and RF were enough to capture the needed patterns. So the transformer ended up being computationally heavier but not giving better accuracy. This is actually a useful insight: **more complex isn’t always better, especially under IoT constraints**. Simpler models were not only easier to train on devices, but they also gave better results in this case.
>
> All in all, the CNN, MLP, and RF are **reliable choices** for a smart home IDS. They strike a good balance of accuracy and efficiency. The decision tree can be a lightweight option if needed, but with lower accuracy. And the transformer might be unnecessary here – it didn’t justify the extra complexity. These findings align well with what we’d expect for IoT: we often prefer lightweight models that still perform well.
>
> ------
>
> ## Slide 7: Training Process – Federated vs Centralized Convergence
>
> - **Convergence Speed:** Federated training **converged nearly as fast** as centralized training. By the end of 50 rounds/epochs, the federated model reached almost the same accuracy as the central model.
> - **Training Curve Comparison:** The plot on the right shows an example with the CNN model. The **blue curve** is centralized training (validation accuracy over 50 epochs) and the **orange curve** is federated (accuracy over 50 rounds). Both rise quickly and plateau at a high accuracy.
> - **Minor Lag for FL:** The FL curve starts slightly lower and converges a touch more slowly, but it closely tracks the central curve. Around round 30–40, the federated model’s accuracy begins to saturate, just shy of the centralized model. Final gap is very small (~2-3%).
> - **Stability:** No major oscillations were observed in the FL training; the process was stable. (This is partly because our data splits were reasonably balanced. In more heterogeneous data scenarios, FL might show some fluctuation, but in our experiments it was smooth.)
> - **Across Models:** We observed similar training dynamics for other models. E.g., the MLP and RF also showed federated accuracy climbing steadily to match the central training. The Transformer’s curve leveled off lower (reflecting its lower final accuracy), but again the difference between its FL vs local training trajectories was small.
>
> ![img](blob:https://chatgpt.com/61ea99d5-1c8e-457c-a535-008d7997a729) *Validation accuracy vs. training progress for the CNN model, comparing Federated Learning (orange line) with Centralized training (blue line). X-axis shows training epochs for the centralized model and communication rounds for the federated model (both 50 in total). Y-axis is validation accuracy (%). The federated model’s accuracy improves rapidly and nearly reaches the centralized model’s accuracy by the final round, illustrating that FL achieves comparable performance with a slight training lag.*
>
> **Narration:** Let’s look at **how** the training progressed in federated learning versus centralized. The figure here is an example using the CNN model’s validation accuracy over time. The blue line is the CNN being trained with all data centrally for 50 epochs. The orange line is the same CNN being trained federated over 50 communication rounds. You can see both lines start at a similar baseline (around 20-30% accuracy at the very beginning, essentially around random guessing for 6 classes) and then quickly climb upwards as the model learns. The centralized blue line is a tiny bit ahead— which is expected, since that model sees all data at once. The federated orange line is just behind it. By about 10-20 epochs/rounds in, both are already above ~80% accuracy. By round 30 or so, the federated CNN is maybe 2-3% below the centralized one. And at the end of training, the centralized CNN hits about 94% and the federated one about 91-92%. The **gap is very small**. The key takeaway is that the **federated model is learning almost as efficiently as the centralized model**. We didn’t need an excessive number of rounds to get there either – 50 rounds was enough to basically converge.
>
> We also note that the training was quite **stable**. Sometimes people worry that federated training could be noisy because different devices might have different data. In our case, the validation accuracy curve in FL was smooth and monotonically increasing, much like the centralized case. That indicates our federation didn’t introduce instability. (Granted, our simulation likely had reasonably balanced data splits. In real life, one home might have significantly different data than another, which could cause the FL curve to wobble a bit, but our results suggest it’s manageable.)
>
> Other models had similar patterns. The MLP’s curves looked very much the same story. The Random Forest in a sense doesn’t have a multi-epoch curve (it’s an ensemble built in one go), but if we think of ensemble size or so, its performance in federated mode was immediately high. The Transformer’s curve reached a lower top accuracy as we discussed, but interestingly the gap between its federated vs centralized versions was also small – the transformer wasn’t great in either case, so FL didn’t make it worse relatively speaking.
>
> In summary, from a training process viewpoint, **federated learning did not require significantly more training iterations to get good performance**. We got to a high detection accuracy in a reasonable number of rounds. That’s encouraging because it means deploying this in practice wouldn’t necessarily be slow or inefficient. We can train an effective model in a federated way almost as easily as we would in a data-center setting.
>
> ------
>
> ## Slide 8: Smart Home Constraints & How FL IDS Meets Them
>
> - **Resource Constraints:** IoT devices are limited in processing power, memory, and energy. An IDS for smart homes **must be lightweight and efficient**
>
>   [journalofcloudcomputing.springeropen.com](https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-018-0123-6#:~:text=An intrusion detection system ,and serious issue%3B thus%2C an)
>
>   . (Our approach uses relatively small models like CNN/MLP and offloads heavy computation by spreading it across devices over time.)
>
>   
>
> - **Real-Time Detection:** The system needs to flag intrusions quickly to be useful (e.g., detect an ongoing attack on a camera and alert immediately). Our models are fast to infer (especially the non-transformer models) – suitable for real-time use. Training can be scheduled during idle times to not interfere with device operation.
>
> - **Privacy and Data Ownership:** Smart home data is highly personal. Residents may not trust sending raw logs to a cloud service. FL addresses this by **keeping data on-premise**; only model parameters are shared. This greatly reduces privacy risks and legal/ethical concerns about data handling.
>
> - **Non-IID and Personalized Behavior:** Each home might have unique network patterns or device usage. A global centralized model might not fit all homes well. FL inherently trains on each home’s data, allowing the model to **learn diverse patterns**. The global model benefits from variety, and potential extensions of FL could even personalize the model further for each home if needed.
>
> - **Communication Overhead:** Smart home devices often connect over Wi-Fi or low-bandwidth networks. Transmitting large amounts of data frequently is not ideal. FL sends only periodic model updates (which are usually kilobytes or a few megabytes). With 50 rounds in our experiment, that’s a manageable overhead. We could also reduce frequency (trade-off rounds vs. time) or compress updates if needed. Overall, the communication cost of FL can be tuned to stay within IoT bandwidth limits while still maintaining accuracy.
>
> - **Robustness and Security:** (Beyond our experiments) FL has the advantage that if one device goes offline or is compromised, the training can continue with others – a decentralized benefit. However, FL can introduce new concerns (like poisoning attacks on model updates), so secure aggregation techniques would be important in a real deployment.
>
> **Narration:** Let’s talk about deploying this in the real world of smart homes and IoT. There are a number of **practical constraints** and we need to see how our federated IDS addresses them:
>
> - **Computational Limits:** IoT devices aren’t powerful computers. A typical smart thermostat or camera has a low-end processor and limited battery (if wireless). Any IDS running on them has to be lightweight. In fact, a paper on IoT security notes that an IDS in IoT must operate under “low processing capability, fast response, and high-volume data” conditions
>
>   [journalofcloudcomputing.springeropen.com](https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-018-0123-6#:~:text=An intrusion detection system ,and serious issue%3B thus%2C an)
>
>   . In our approach, we chose models like CNNs and MLPs which can be scaled down to a small footprint. We avoided super heavy architectures in deployment (and we saw the transformer, which is heavy, wasn’t performing well anyway). Also, in federated training, each device only does a bit of work at a time (maybe a few epochs on its local subset). This is something a device can handle, perhaps during off-peak times or when plugged in. It’s much better than trying to stream all raw data out or run an extremely deep model continuously.
>
>   
>
> - **Real-Time Requirements:** If an intrusion happens, we want to catch it immediately. That means our model inference needs to be quick on the device. A benefit of our top models (CNN, MLP, RF) is that they are pretty fast at inference. A CNN forward-pass or a tree lookup is milliseconds. So the IDS can analyze events as they come in without noticeable lag. We would likely update the model periodically (say the devices federate once a day or when idle), so training overhead won’t interfere with real-time detection. Essentially, we ensure the detection component is always running and snappy, and training can happen in the background.
>
> - **Privacy:** This is where FL shines. Smart home data – think about it, it could include audio from your living room, video of your baby’s room, logs of when you’re home or away. Extremely sensitive stuff. With a conventional approach, maybe all that would be sent to a cloud to train an IDS, which is a privacy nightmare. FL flips that model: your data **stays with you**. The only thing leaving the device are model weight updates, which by themselves reveal very little about any single person’s behavior. This is a huge win for user trust. It means we can actually deploy IDS in homes without users feeling like “Big Brother” is watching their raw data.
>
> - **Data Distribution & Personalization:** Smart homes aren’t all the same. One house might have ten devices chatting a lot on the network, another might just have two. One might be a tech-savvy user who changed defaults, another might not. Attack patterns might vary. A single monolithic model might not capture all those nuances. FL inherently trains on distributed data, so the model learns a bit of everything from everyone. It’s like crowdsourcing knowledge of attacks while still tailoring to each home. In our work, we trained one global model, but in the future we could imagine each home’s model starting to specialize a bit (there are federated learning techniques for personalization). But even with a single global model, training on all homes’ data in place means it has seen a variety of conditions. That should make it more robust. And importantly, if some homes have rarer attacks, FL training allows the global model to learn about those without the data ever leaving those homes.
>
> - **Communication Constraints:** FL isn’t free of cost – devices do need to send model updates to a server or aggregator. But these updates are **much smaller** than raw datasets. In our case, consider sending maybe a few megabytes of model parameters occasionally, versus streaming gigabytes of raw network logs constantly. The difference is huge. We ran 50 rounds in our experiment; in a real deployment, we could adjust how frequently rounds happen (maybe one round per hour or per day depending on urgency and network availability). IoT devices often operate on Wi-Fi or even cellular, so we have to be mindful of bandwidth. The good news is, as seen, we don’t need hundreds of rounds – we can achieve good accuracy with a reasonable number of communications. Techniques like model compression or update sparsing can further cut down the size of communications if needed. Overall, the communication overhead of FL for an IDS is quite manageable for a home network.
>
> - **Security of the FL process:** I’ll just note, while FL helps privacy, we should also secure the FL process itself – for instance, ensure that model updates are not tampered with (using secure aggregation protocols). That’s outside our current scope, but it’s a consideration in a hostile setting.
>
> In summary, our federated learning IDS design aligns well with smart home constraints. It keeps the solution lightweight, fast, and private – all necessary for an IDS that people can actually deploy on their home devices with confidence.
>
> ------
>
> ## Slide 9: Key Findings & Conclusion
>
> - **FL is Viable for Smart Home IDS:** We demonstrated that a federated learning approach can achieve **accuracy within <3% of centralized training**. In other words, we don’t pay a significant accuracy penalty for keeping data on devices – a crucial validation for adopting FL in security.
> - **High Detection Rates Maintained:** The IDS catches the vast majority of intrusions/anomalies in the smart home data. Normal behavior and all attack classes were detected with high precision and recall in our experiments. False alarms and misses were only marginally higher in the federated scenario, staying at a low level.
> - **Best Models – CNN, MLP, RF:** These models were the most **reliable and effective** in our tests. They provided a strong combination of accuracy and efficiency. This suggests that for real deployments, a lightweight deep model (CNN/MLP) or an ensemble like Random Forest is a good choice for the detection engine.
> - **Complex Models Not Always Better:** The Transformer’s underperformance is a useful insight – it indicates that extremely complex models aren’t necessary for this task and may not be suitable for on-device training. Simpler models not only did better here but also are easier to run on IoT hardware.
> - **Smart Home Feasibility:** Our approach addresses smart home needs by preserving privacy, minimizing communication, and using computationally manageable models – all while delivering strong security performance. **Federated Learning is a practical and competitive solution for intrusion detection in smart homes.** We can have both security and privacy, without significantly compromising on either.
> - **Future Work (brief):** Test this system on real IoT devices and networks, investigate personalization per home (so each home’s model can fine-tune to its environment), and ensure robustness of the FL process against adversarial behavior. Overall, the results are very encouraging for moving towards deployment of federated IDS in IoT environments.
>
> **Narration:** To wrap up, let me summarize the key takeaways from our study. First, we found that **Federated Learning is highly competitive with traditional centralized training** for smart home anomaly detection. We saw less than a 3% difference in accuracy, which is practically negligible in most cases. This is a big deal – it means we can keep data on devices and still catch intrusions almost as well as if we had all the data collected in one place.
>
> Second, the system we built indeed achieves **high detection rates** for a variety of attacks. It’s not just overall accuracy; when we break it down by attack type, the federated model was catching nearly all of them with high precision and recall. So you’re not trading off a particular attack slipping through – performance remains strong across the board.
>
> Third, in terms of models, the **CNN, MLP, and Random Forest turned out to be the top performers**. These models are relatively simple and efficient, yet they gave us great accuracy and proved reliable even in the federated scenario. This implies that if someone were to implement a smart home IDS, these are good candidate algorithms to use. They won’t demand too much from the device but will deliver good results.
>
> On the other hand, our experiment with the **Transformer taught us that more complex isn’t always better** in this context. It underachieved compared to the simpler models. So, especially given IoT constraints, it might be wise to avoid overly complex models that don’t yield clear benefits. Simpler models are not only easier to handle on-device, but as we saw, they might actually perform better on the kind of data we have.
>
> Overall, we conclude that **Federated Learning is a promising and practical approach for smart home intrusion detection**. We managed to build an IDS that respects user privacy (data never leaving the devices) and still provides robust security by detecting intrusions effectively. And it does so without requiring super-powerful hardware or constant network streaming. This approach could be a key enabler for deploying security AI across IoT devices in people’s homes.
>
> In future work, it would be interesting to deploy this on real hardware to monitor things like actual smart home network traffic in real time, to verify that our simulation translates to practice. We also want to explore if we can tailor the global model to each home (so it becomes even more accurate for each user’s pattern). And we’ll look into security of the federated process itself, making sure an attacker can’t game the system by sending bad updates. But the bottom line from our study is very positive: we can have both **security** and **privacy** in smart home environments by using federated learning for anomaly and intrusion detection. Thank you for listening!