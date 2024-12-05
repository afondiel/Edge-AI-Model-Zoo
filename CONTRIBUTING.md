# **Contribution Guidelines**

**Welcome to Awesome Smol Models!** We’re excited to have you contribute. Please follow these steps to ensure smooth collaboration.  

## **How to Contribute**  
1. **Find an area to contribute**:  
   - Add a new model.  
   - Enhance documentation or examples.  
   - Suggest new tools/resources.  
2. **Fork the repository**: Click the “Fork” button to create a copy of this repo in your GitHub account.  
3. **Clone your fork**:  
   ```bash  
   git clone https://github.com/afondiel/awesome-smol-models  
   cd awesome-smol-models  
   ```  
4. **Create a branch**:  
   ```bash  
   git checkout -b <branch-name>  
   ```  
5. **Make your changes**: Add or edit models, update files, or fix issues.  
6. **Run a check**: Validate formatting and links using tools like `markdownlint` or `awesome-lint`.  
7. **Submit a Pull Request**: Push your branch and create a PR following the template below.  

## **Pull Request Template**  
```  
# Pull Request  

## Description  
Provide a summary of your changes, including the type of contribution (new model, documentation, fix, etc.).  

## Checklist  
- [ ] My code follows the repository style guide.  
- [ ] I have added appropriate references and links.  
- [ ] I have tested my changes locally.  

## Related Issue  
If applicable, link the issue number: `#<issue-number>`  

## Additional Notes  
Any other relevant information.  
```  

## **Add New Models**

Add the model under the relevant section with the following format:

```markdown  
| TinyYOLO          | Object Detection      | Mobile, Edge         | [GitHub](https://github.com)     |  
```

If you manage to run the model on any edge device/sim tool, you're welcome to add its `benchmarks` to highlight `performance` and `usability`.  

### **Benchmark Table Format**  
| **Model**          | **Task**             | **Accuracy**         | **Latency (ms)** | **Model Size (MB)** | **Platform**         | **References**                   |  
|---------------------|----------------------|----------------------|------------------|---------------------|----------------------|-----------------------------------|  
| MobileNet V2        | Image Classification | 72.0%                | 25               | 4.3                 | Android, iOS, Web    | [TensorFlow Lite](https://www.tensorflow.org/lite) |  

### **How to Add Benchmarks**

1. **Download and Evaluate**: Run the model on a standardized dataset (e.g., ImageNet, COCO).  
2. **Measure Performance**:  
   - Use tools like ONNX Runtime, TensorFlow Lite Benchmark Tool, or CoreML Profiler.  
   - Record latency on mobile/edge devices (e.g., Raspberry Pi, Jetson Nano).  
3. **Document Results**: Use the table above to document key metrics.  
4. **Submit with PR**: Attach the benchmark results with your PR submission.  

## **Enhancing Resources**  
- Propose new blog posts, podcasts, or tutorials.  
- Validate all links and content for accuracy.

## **Examples of a Good Contribution**  
**Adding a New Model**  
- Fork the repository.  
- Add the model under the relevant section with the following details:  
  ```markdown  
  | TinyYOLO          | Object Detection      | Mobile, Edge         | [GitHub](https://github.com)     |  
  ```  
- Run benchmarks and attach results.  
- Submit a pull request. 
