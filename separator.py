import inference

class Separate():
    def __init__(self, config='config/default.yaml', embedder='embedder.pt', checkpoint='voiceSplit-trained-with-Si-SRN-GE2E-CorintinJ-best_checkpoint.pt'):
        self.config=config
        self.embedder=embedder
        self.checkpoint=checkpoint

    def one(self, path, reference, output_dir, output):
        inference.wrap(self.config, self.embedder,
                       self.checkpoint,
                       path, reference, output_dir, output)

    def many(self, paths, references, output_dirs, outputs):
        for path, reference, output_dir, output in zip(paths, references, output_dirs, outputs):
            self.one(path, reference, output_dir, output)


if __name__ == '__main__':

    inference.wrap('config/default.yaml', 'embedder.pt', 'voiceSplit-trained-with-Si-SRN-GE2E-CorintinJ-best_checkpoint.pt',
                   'simahn-cut.wav', 'sim-sample.wav', 'directory_with_japanese_result', 'not_even_even_even_japanese_together.wav')
