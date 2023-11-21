from decimal import Decimal

class ArithmeticEncoding:
    """
    ArithmeticEncoding is a class for building arithmetic encoding.
    """

    def __init__(self, frequency_table):
        self.probability_table = self.get_probability_table(frequency_table)

    def get_probability_table(self, frequency_table):
        """
        Calculates the probability table out of the frequency table.
        """
        total_frequency = sum(list(frequency_table.values()))

        probability_table = {}
        for key, value in frequency_table.items():
            probability_table[key] = value/total_frequency

        return probability_table

    def get_encoded_value(self, encoder):
        """
        After encoding the entire message, this method returns the single value that represents the entire message.
        """
        last_stage = list(encoder[-1].values())
        last_stage_values = []
        for sublist in last_stage:
            for element in sublist:
                last_stage_values.append(element)

        last_stage_min = min(last_stage_values)
        last_stage_max = max(last_stage_values)

        return (last_stage_min + last_stage_max)/2

    def process_stage(self, probability_table, stage_min, stage_max):
        """
        Processing a stage in the encoding/decoding process.
        """
        stage_probs = {}
        stage_domain = stage_max - stage_min
        for term_idx in range(len(probability_table.items())):
            term = list(probability_table.keys())[term_idx]
            term_prob = Decimal(probability_table[term])
            cum_prob = term_prob * stage_domain + stage_min
            stage_probs[term] = [stage_min, cum_prob]
            stage_min = cum_prob
        return stage_probs

    def encode(self, msg, probability_table):
        """
        Encodes a message.
        """

        encoder = []

        stage_min = Decimal(0.0)
        stage_max = Decimal(1.0)

        for msg_term_idx in range(len(msg)):
            stage_probs = self.process_stage(probability_table, stage_min, stage_max)

            msg_term = msg[msg_term_idx]
            stage_min = stage_probs[msg_term][0]
            stage_max = stage_probs[msg_term][1]

            encoder.append(stage_probs)

        stage_probs = self.process_stage(probability_table, stage_min, stage_max)
        encoder.append(stage_probs)

        encoded_msg = self.get_encoded_value(encoder)

        return encoder, encoded_msg

    def decode(self, encoded_msg, msg_length, probability_table):
        """
        Decodes a message.
        """

        decoder = []
        decoded_msg = ""

        stage_min = Decimal(0.0)
        stage_max = Decimal(1.0)

        for idx in range(msg_length):
            stage_probs = self.process_stage(probability_table, stage_min, stage_max)

            for msg_term, value in stage_probs.items():
                if encoded_msg >= value[0] and encoded_msg <= value[1]:
                    break

            decoded_msg = decoded_msg + msg_term
            stage_min = stage_probs[msg_term][0]
            stage_max = stage_probs[msg_term][1]

            decoder.append(stage_probs)

        stage_probs = self.process_stage(probability_table, stage_min, stage_max)
        decoder.append(stage_probs)

        return decoder, decoded_msg
    
import numpy as np
from collections import Counter
import heapq
import torch
from bitarray import bitarray

class Node:
    def __init__(self):
        self.value = None
        self.children = {}


class Trie:
    def __init__(self):
        self.root = Node()

    def add(self, code, value):
        node = self.root
        for bit in code:
            if bit not in node.children:
                node.children[bit] = Node()
            node = node.children[bit]
        node.value = value

    def find(self, code):
        node = self.root
        for bit in code:
            if bit not in node.children:
                return None
            node = node.children[bit]
        return node.value

def huffman_coding(tensor):
    
    # check if quint8 , else it can not flatten
    if tensor.dtype == torch.quint8:
        tensor = tensor.dequantize()
        
    # Flatten the tensor and convert it to a list
    flattened_tensor = tensor.flatten().tolist()
    
    # Count the frequency of each value in the tensor
    frequency = Counter(flattened_tensor)
    
    # Create nodes for each value-frequency pair
    nodes = [[weight, [value, '']] for value, weight in frequency.items()]
    
    # Build the Huffman tree
    heapq.heapify(nodes)
    while len(nodes) > 1:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
        
        for pair in left[1:]:
            pair[1] = '0' + pair[1]
        for pair in right[1:]:
            pair[1] = '1' + pair[1]
        
        heapq.heappush(nodes, [left[0] + right[0]] + left[1:] + right[1:])
    
    # Generate the Huffman codes for each value
    huffman_codes = {}
    for pair in nodes[0][1:]:
        # huffman_codes[pair[0]] = pair[1]
        huffman_codes[round(pair[0], 5)] = pair[1]
        
    
        
    # Initialize an empty bitarray
    bits = bitarray()
    
    # Extend the bits based on Huffman codes
    for val in flattened_tensor: # this part is time consuming
        bits.extend(map(int, huffman_codes[round(val, 5)])) # to avoid precision issues
        
    # Convert the bitarray to bytes
    byte_representation = bits.tobytes()
    
    
    # # rounded_tensor = np.round(your_tensor, 6)
    # encoded_data = ''.join([huffman_codes[val] for val in flattened_tensor])
    
    # # Convert the string to its integer representation
    # int_representation = int(encoded_data, 2)
    # # Pack the integer into bytes using the appropriate number of bytes
    # num_bytes = (len(encoded_data) + 7) // 8  # Calculate the number of bytes required
    # byte_representation = int_representation.to_bytes(num_bytes, byteorder='big')
    # bits = np.array([int(x) for x in byte_representation], dtype=np.uint8)
    tensor_shape = tensor.shape
        

    
    return huffman_codes, byte_representation, tensor_shape



def huffman_decode(encoded_data, huffman_codes, tensor_shape):
    
    trie = Trie()
    for code, value in huffman_codes.items():
        trie.add(value, code)

    decoded_data =[]
    current_code = ""
    for byte in encoded_data:
        for i in range(8):
            current_bit = (byte >> (7 - i)) & 1
            current_code += str(current_bit)
    
            value = trie.find(current_code)
            if value is not None:
                decoded_data.append(value)
                current_code = ""
    
    # '''
    
    
    # Reshape the decoded data into the original tensor shape
    if len(decoded_data) != np.prod(tensor_shape): # to handle bit to byte manipulation as it add some zeros
        original_tensor = np.array(decoded_data[:np.prod(tensor_shape)]).reshape(tensor_shape)
    else:
        original_tensor = np.array(decoded_data).reshape(tensor_shape)
    return original_tensor



def huffman_decode2(encoded_data, huffman_codes, tensor_shape):
    # Invert the Huffman codes
    inverted_codes = {code: value for value, code in huffman_codes.items()}
    
    # encoded_data = ''.join(format(byte, '08b') for byte in encoded_data)
    
    # Decode the encoded data using the inverted Huffman codes
    # decoded_data = ''
    decoded_data =[]
    current_code = ''
    # for bit in encoded_data:
    #     current_code += bit
    #     if current_code in inverted_codes:
    #         # decoded_data += ','+str(inverted_codes[current_code])
    #         decoded_data.append(inverted_codes[current_code])
    #         current_code = ''
    # '''
    # Extracting bits for decoding
    for byte in encoded_data: # maybe using a tree is faster than this dic data
        for i in range(8):  # 8 bits in a byte
            current_bit = (byte >> (7 - i)) & 1  # Extract each bit
            current_code += str(current_bit)
    
            if current_code in inverted_codes:
                decoded_data.append(inverted_codes[current_code])
                # current_code1 = current_code
                current_code = ''
    
    # '''
    
    
    # Reshape the decoded data into the original tensor shape
    if len(decoded_data) != np.prod(tensor_shape): # to handle bit to byte manipulation as it add some zeros
        original_tensor = np.array(decoded_data[:np.prod(tensor_shape)]).reshape(tensor_shape)
    else:
        original_tensor = np.array(decoded_data).reshape(tensor_shape)
    return original_tensor

def huffman_decode1(encoded_data, huffman_codes, tensor_shape):
    # Convert float keys to strings
    root = {str(key): value for key, value in huffman_codes.items()}
    
    # Then call the build_trie function
    build_trie(huffman_codes)
    decoded_data = []

    current_node = root
    for byte in encoded_data:
        for i in range(7, -1, -1):  # Loop through bits in byte
            bit = (byte >> i) & 1
            if str(bit) not in current_node.children:
                break
            current_node = current_node.children[str(bit)]
            if current_node.value is not None:
                decoded_data.append(current_node.value)
                current_node = root
    original_tensor = np.array(decoded_data[:-1]).reshape(tensor_shape)
    return original_tensor



# # Decoding the Huffman-coded data back to the original tensor
# decoded_tensor = huffman_decode(encoded_data, huffman_result)
# print(your_tensor)
# print(decoded_tensor)