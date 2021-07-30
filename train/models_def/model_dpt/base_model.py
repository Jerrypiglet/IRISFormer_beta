import torch


class BaseModel(torch.nn.Module):
    def load(self, path, skip_keys=[], keep_keys=[]):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        strict = True
        if skip_keys!=[]:
            # print('===', len(parameters.keys()), skip_keys, [k for k, v in parameters.items()])
            parameters = {k:v for k, v in parameters.items() if not any([x in k for x in skip_keys])}
            print('==>', len(parameters.keys()))
            strict = False

        if keep_keys!=[]:
            # print('===', len(parameters.keys()), keep_keys, [k for k, v in parameters.items()])
            parameters = {k:v for k, v in parameters.items() if any([x in k for x in keep_keys])}
            print('== ONLY restore at DPT: ==>', len(parameters.keys()))
            strict = False

        self.load_state_dict(parameters, strict=strict)
