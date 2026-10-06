function values = rand_mt19937(seed, count)

if ~isnumeric(seed) || ~isreal(seed) || ~isscalar(seed) || ~isfinite(seed) || ...
        seed < 0 || seed > 4294967295 || seed ~= fix(seed)
    error('NeuroFANN:InvalidParameter', 'Seed must be an integer in the uint32 range.');
end
if ~isnumeric(count) || ~isreal(count) || ~isscalar(count) || ~isfinite(count) || ...
        count < 0 || count ~= fix(count)
    error('NeuroFANN:InvalidParameter', 'The number of draws must be a nonnegative integer.');
end
seed = double(seed);
count = double(count);
if seed == 0
    seed = 5489;
end
state = zeros(624, 1);
state(1) = seed;
for index = 2:624
    previous = state(index - 1);
    mixed = bitxor(previous, floor(previous / 1073741824));
    state(index) = mod(multiplyModulo(mixed) + (index - 1), 4294967296);
end
total = 2 * count;
words = zeros(total, 1);
position = 625;
produced = 0;
while produced < total
    if position > 624
        state = twist(state);
        position = 1;
    end
    taken = min(625 - position, total - produced);
    words(produced + (1:taken)) = state(position + (0:taken - 1));
    position = position + taken;
    produced = produced + taken;
end
words = bitxor(words, floor(words / 2048));
words = bitxor(words, bitand(words * 128, 2636928640));
words = bitxor(words, bitand(words * 32768, 4022730752));
words = bitxor(words, floor(words / 262144));
high = floor(words(1:2:total) / 32);
low = floor(words(2:2:total) / 64);
values = (high * 67108864 + low) / 9007199254740992;

end

function product = multiplyModulo(value)

high = floor(value / 65536);
low = value - high * 65536;
product = mod(mod(1812433253 * high, 65536) * 65536 + 1812433253 * low, 4294967296);

end

function state = twist(state)

index = (1:227).';
state(index) = mixWords(state(index), state(index + 1), state(index + 397));
index = (228:454).';
state(index) = mixWords(state(index), state(index + 1), state(index - 227));
index = (455:623).';
state(index) = mixWords(state(index), state(index + 1), state(index - 227));
state(624) = mixWords(state(624), state(1), state(397));

end

function mixed = mixWords(current, following, distant)

combined = (current >= 2147483648) * 2147483648 + mod(following, 2147483648);
mixed = bitxor(bitxor(distant, floor(combined / 2)), mod(combined, 2) * 2567483615);

end
