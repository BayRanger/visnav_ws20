/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <visnav/common_types.h>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

namespace visnav {

class BowDatabase {
 public:
  BowDatabase() {}

  inline void insert(const FrameCamId& fcid, const BowVector& bow_vector) {

  /* ------------------------------ TASK SHEET 3 ------------------------------ */
    for (const auto& b : bow_vector)
      if (inverted_index.find(b.first) != inverted_index.end())
        inverted_index[b.first].emplace_back(fcid, b.second);
      else
        inverted_index[b.first] = {std::make_pair(fcid, b.second)};
  /* -------------------------------------------------------------------------- */
  
  }

  inline void query(const BowVector& bow_vector, size_t num_results,
                    BowQueryResult& results) const {

    /* ------------------------------ TASK SHEET 3 ------------------------------ */
    results.clear();
    results.reserve(num_results);

    std::unordered_map<FrameCamId, std::vector<std::pair<WordId, WordValue>>>
        relevants;

    for (const auto& b : bow_vector) {
      const auto& query_word_id = b.first;
      if (inverted_index.find(query_word_id) != inverted_index.end())
        for (const auto& kv : inverted_index.at(query_word_id)) {
          const auto& frame_id = kv.first;
          const auto& word_value = kv.second;
          if (relevants.find(frame_id) != relevants.end())
            relevants[frame_id].push_back(
                std::make_pair(query_word_id, word_value));
          else
            relevants[frame_id] = {std::make_pair(query_word_id, word_value)};
        }
    }

    std::vector<std::pair<FrameCamId, double>> candidates;
    for (const auto& relevant : relevants) {
      const auto& word_pairs = relevant.second;

      // In case of there is any duplicated word
      std::unordered_map<WordId, double> db_vector;
      for (const auto& p : word_pairs) {
        const auto& word_id = p.first;
        const auto& word_value = p.second;
        if (db_vector.find(word_id) != db_vector.end())
          db_vector[word_id] += word_value;
        else
          db_vector[word_id] = word_value;
      }

      double dist = 2;
      for (const auto& word_pair : bow_vector) {
        const auto& word_id = word_pair.first;
        const auto& word_value = word_pair.second;
        if (db_vector.find(word_id) != db_vector.end())
          dist += abs(word_value - db_vector[word_id]) - abs(word_value) -
                  abs(db_vector[word_id]);
      }

      candidates.emplace_back(relevant.first, dist);
    }

    num_results = std::min(num_results, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + num_results,
                      candidates.end(),
                      [](auto& l, auto& r) { return l.second < r.second; });
    for (size_t i = 0; i < num_results; i++) results.push_back(candidates[i]);
  /* -------------------------------------------------------------------------- */

  }

  void clear() { inverted_index.clear(); }

 protected:
  tbb::concurrent_unordered_map<
      WordId, tbb::concurrent_vector<std::pair<FrameCamId, WordValue>>>
      inverted_index;
};

}  // namespace visnav
