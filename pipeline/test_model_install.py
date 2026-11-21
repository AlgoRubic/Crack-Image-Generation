import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from download_models import ROOT,install_archive


class ModelInstallation(unittest.TestCase):
    def bundle(self,root,content=b'verified model',extra=False):
        name='models/weights/exp1/model.pt'
        manifest=dict(checkpoints=[dict(path=name,size_bytes=len(content),sha256=hashlib.sha256(content).hexdigest())])
        archive=root/'bundle.zip'
        with zipfile.ZipFile(archive,'x') as stream:
            stream.writestr(name,content);stream.writestr('models/checkpoint_manifest.json',json.dumps(manifest))
            if extra:stream.writestr('../escape.pt',b'bad')
        return archive,manifest,name

    def test_verified_install_and_repeat_preserve_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'experiments') as temp:
            root=Path(temp);archive,manifest,name=self.bundle(root)
            install_archive(archive,root,manifest)
            self.assertEqual((root/name).read_bytes(),b'verified model')
            install_archive(archive,root,manifest)
            self.assertEqual((root/name).read_bytes(),b'verified model')

    def test_existing_different_model_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'experiments') as temp:
            root=Path(temp);archive,manifest,name=self.bundle(root)
            target=root/name;target.parent.mkdir(parents=True);target.write_bytes(b'user model')
            with self.assertRaises(ValueError):install_archive(archive,root,manifest)
            self.assertEqual(target.read_bytes(),b'user model')

    def test_unexpected_archive_member_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'experiments') as temp:
            root=Path(temp);archive,manifest,name=self.bundle(root,extra=True)
            with self.assertRaises(ValueError):install_archive(archive,root,manifest)
            self.assertFalse((root/name).exists())


if __name__=='__main__':unittest.main()
