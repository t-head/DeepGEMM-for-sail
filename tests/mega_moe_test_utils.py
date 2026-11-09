import torch


def check_bf16_offset_views(source, expected, forward, *, replay=None):
    """Compare aligned and offset forwards, including guards around both views."""
    def make_view(offset):
        storage = torch.full((source.numel() + offset + 8,), 13.0,
                             dtype=source.dtype, device=source.device)
        view = storage[offset:offset + source.numel()].view_as(source)
        assert view.is_contiguous() and view.storage_offset() == offset
        return storage, view

    def check_guards(storage, offset):
        assert torch.all(storage[:offset] == 13)
        assert torch.all(storage[offset + source.numel():] == 13)

    # Exercise input-only, output-only and joint offsets, including every BF16
    # alignment modulo 16 bytes.
    for input_offset, output_offset in ((1, 0), (0, 1), *((i, i) for i in range(1, 8))):
        input_storage, offset_input = make_view(input_offset)
        output_storage, offset_output = make_view(output_offset)
        offset_input.copy_(source)
        actual = forward(offset_input, offset_output)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(offset_input, source, rtol=0, atol=0)
        check_guards(input_storage, input_offset)
        check_guards(output_storage, output_offset)

    if replay is not None:
        replay(offset_input, offset_output)
        check_guards(input_storage, input_offset)
        check_guards(output_storage, output_offset)
