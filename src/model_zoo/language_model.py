class LanguageModel():
    def __init__(self, model):
        self.model = model
        
        
    def __str__(self):
       return f"LanguageModel(model={self.model})"
    
    def __repr__(self):
        return str(self)
    
    def config_model(self,**kwargs):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def run(self,**kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

