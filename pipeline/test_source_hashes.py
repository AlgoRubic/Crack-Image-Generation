import tempfile
from pathlib import Path
import unittest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from hash_source_images import ROOT,hash_original


class SourceHashTests(unittest.TestCase):
    def test_metadata_differences_do_not_hide_identical_pixels(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'experiments') as temp:
            a,b=Path(temp)/'a.png',Path(temp)/'b.png'
            image=Image.new('RGB',(16,16),(100,110,120))
            image.save(a)
            metadata=PngInfo();metadata.add_text('test','different encoding metadata')
            image.save(b,pnginfo=metadata)
            first,_=hash_original(a.relative_to(ROOT).as_posix(),{})
            second,_=hash_original(b.relative_to(ROOT).as_posix(),{})
            self.assertNotEqual(first['sha256'],second['sha256'])
            self.assertEqual(first['decoded_rgb_sha256'],second['decoded_rgb_sha256'])
            cached,changed=hash_original(first['original_path'],{first['original_path']:first})
            self.assertFalse(changed)
            self.assertEqual(cached,first)


if __name__=='__main__': unittest.main()
